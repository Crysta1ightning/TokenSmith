import argparse
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, Optional

from src.config import QueryPlanConfig
from src.generator import answer, format_prompt, get_llm_stats, set_model_cache_enabled
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger, RunLogger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Welcome to TokenSmith!"
    )

    # Required arguments
    parser.add_argument(
        "mode",
        choices=["index", "chat"],
        help="operation mode: 'index' to build index, 'chat' to query"
    )

    # Common arguments
    parser.add_argument(
        "--pdf_dir",
        default="data/chapters/",
        help="directory containing PDF files (default: %(default)s)"
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="prefix for generated index files (default: %(default)s)"
    )
    parser.add_argument(
        "--model_path",
        help="path to generation model (uses config default if not specified)"
    )
    parser.add_argument(
        "--system_prompt_mode",
        choices=["baseline", "tutor", "concise", "detailed"],
        default="baseline",
        help="system prompt mode (choices: baseline, tutor, concise, detailed)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="deterministic decoding: temp=0, top_k=1, top_p=1, seed=0"
    )
    parser.add_argument(
        "--no_cache_model",
        action="store_true",
        help="disable in-process model cache; force reload each question"
    )
    # Context display options (chat)
    parser.add_argument(
        "--show_context",
        action="store_true",
        help="print retrieved context chunks and sources after each answer"
    )
    parser.add_argument(
        "--sources_only",
        action="store_true",
        help="print retrieved context only (skip model generation)"
    )
    parser.add_argument(
        "--context_k",
        type=int,
        default=3,
        help="number of retrieved chunks to display when showing context (default: %(default)s)"
    )
    parser.add_argument(
        "--context_chars",
        type=int,
        default=400,
        help="maximum characters per chunk preview when showing context (default: %(default)s)"
    )
    parser.add_argument(
        "--prompt_chunk_chars",
        type=int,
        default=400,
        help="maximum characters per chunk included per chunk in the LLM prompt (default: %(default)s)"
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="prepend a few short exemplars to guide style and citations"
    )
    parser.add_argument(
        "--abstain_threshold",
        type=float,
        default=0.2,
        help="if retrieval confidence is below this, abstain with 'I don't know'"
    )
    parser.add_argument(
        "--score_warp_gamma",
        type=float,
        default=2.5,
        help="nonlinear warp gamma for bounded scores in [0,1]; higher pushes 0.6 closer to 1 (default: %(default)s)"
    )
    
    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--pdf_range",
        metavar="START-END",
        help="specific range of PDFs to index (e.g., '27-33')"
    )
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--visualize",
        action="store_true",
        help="generate visualizations during indexing"
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: QueryPlanConfig):
    """Handles the logic for building the index."""

    # Robust range filtering
    try:
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end + 1)] # Inclusive range
            print(f"Indexing PDFs in range: {start}-{end}")
        else:
            pdf_paths = None
    except ValueError:
        print(f"ERROR: Invalid format for --pdf_range. Expected 'start-end', but got '{args.pdf_range}'.")
        sys.exit(1)
    
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    
    artifacts_dir = cfg.make_artifacts_directory()

    build_index(
        markdown_file="data/book_without_image.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
    )


def get_answer(
    question: str,
    cfg: QueryPlanConfig,
    args: argparse.Namespace,
    logger: "RunLogger",
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None
) -> tuple[str, Dict[str, float], list]:
    """
    Run a single query through the pipeline.
    """    
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]
    
    logger.log_query_start(question)
    
    # Step 1: Get chunks (golden, retrieved, or none)
    retrieval_ms = 0.0
    ranking_ms = 0.0
    prompt_ms = 0.0
    generation_ms = 0.0
    tokens_generated = 0
    confidence = 0.0

    context_items: list = []

    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
        confidence = 1.0
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
        confidence = 0.0
    else:
        # Step 1: Retrieval
        pool_n = max(cfg.pool_size, cfg.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        t0 = time.perf_counter()
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(question, pool_n, chunks)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        # TODO: Fix retrieval logging.
        
        # Step 2: Ranking
        t1 = time.perf_counter()
        ordered = ranker.rank(raw_scores=raw_scores)
        topk_idxs = apply_seg_filter(cfg, chunks, ordered)
        ranking_ms = (time.perf_counter() - t1) * 1000.0
        logger.log_chunks_used(topk_idxs, chunks, sources)
        
        ranked_chunks = [chunks[i] for i in topk_idxs]
        # Compute confidence emphasizing top1 strength with a small support bonus
        confidence = 0.0
        try:
            weights = cfg.ranker_weights or {"faiss": 0.5, "bm25": 0.5}
            w_f = float(weights.get("faiss", 0.0))
            w_b = float(weights.get("bm25", 0.0))
            w_sum = (w_f + w_b) or 1.0

            # Build fused absolute scores per candidate for first m = top_k items
            m = max(1, int(cfg.top_k))
            k_bm25 = 10.0
            gamma = float(getattr(args, "score_warp_gamma", 2.5))
            def _warp(x: float) -> float:
                # Map [0,1] -> [0,1], monotonic; 0.6 -> closer to 1 as gamma increases
                x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                return 1.0 - (1.0 - x) ** max(1.0, gamma)
            fused = []
            for i in ordered[:m]:
                # FAISS abs in (0,1]; default 0 if missing
                s_f = float(raw_scores.get("faiss", {}).get(i, 0.0))
                s_f = max(0.0, min(1.0, s_f))
                # BM25 saturation s/(s+k)
                s_b_raw = float(raw_scores.get("bm25", {}).get(i, 0.0))
                s_b = (s_b_raw / (s_b_raw + k_bm25)) if s_b_raw > 0 else 0.0
                # Apply nonlinear warp to both bounded scores before fusing
                s_f_w = _warp(s_f)
                s_b_w = _warp(s_b)
                fused_i = (w_f * s_f_w + w_b * s_b_w) / w_sum
                fused.append(fused_i)

            if fused:
                top1 = fused[0]
                if len(fused) > 1:
                    mean_others = sum(fused[1:]) / (len(fused) - 1)
                else:
                    mean_others = 0.0
                support_weight = 0.2  # small plus from supporting evidence
                confidence = max(0.0, min(1.0, top1 + support_weight * mean_others))
            else:
                confidence = 0.0
        except Exception:
            confidence = 0.0
        # Prepare context preview items (with scores used for confidence)
        k = max(1, int(getattr(args, "context_k", 3)))
        maxc = max(0, int(getattr(args, "context_chars", 400)))
        faiss_scores_map = raw_scores.get("faiss", {})
        bm25_scores_map = raw_scores.get("bm25", {})
        w_f = float(getattr(cfg, "ranker_weights", {}).get("faiss", 0.0) if hasattr(cfg, "ranker_weights") else 0.5)
        w_b = float(getattr(cfg, "ranker_weights", {}).get("bm25", 0.0) if hasattr(cfg, "ranker_weights") else 0.5)
        w_sum = (w_f + w_b) or 1.0
        k_bm25 = 10.0
        gamma = float(getattr(args, "score_warp_gamma", 2.5))
        def _warp(x: float) -> float:
            x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
            return 1.0 - (1.0 - x) ** max(1.0, gamma)

        for rank, i in enumerate(topk_idxs[:k], start=1):
            preview = (chunks[i] or "")[:maxc]
            s_f_abs = float(faiss_scores_map.get(i, 0.0)) if faiss_scores_map else 0.0
            if s_f_abs < 0.0:
                s_f_abs = 0.0
            if s_f_abs > 1.0:
                s_f_abs = 1.0
            s_b_raw = float(bm25_scores_map.get(i, 0.0)) if bm25_scores_map else 0.0
            s_b_sat = (s_b_raw / (s_b_raw + k_bm25)) if s_b_raw > 0 else 0.0
            fused_abs = (w_f * s_f_abs + w_b * s_b_sat) / w_sum
            fused_warped = (w_f * _warp(s_f_abs) + w_b * _warp(s_b_sat)) / w_sum
            context_items.append({
                "S": rank,
                "index": int(i),
                "source": str(sources[i]),
                "preview": preview,
                "score_faiss": s_f_abs,
                "score_bm25_raw": s_b_raw,
                "score_bm25_sat": s_b_sat,
                "score_fused": fused_abs,
                "score_fused_warped": fused_warped,
            })
        
        # Step 3: Final Re-ranking (if enabled)
        # Disabled till we fix the core pipeline
        # ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.top_k)
    
    # Step 4: Generation
    model_path = args.model_path or cfg.model_path
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode
    # Measure prompt formatting separately (for visibility)
    p0 = time.perf_counter()
    _ = format_prompt(
        ranked_chunks,
        question,
        max_chunk_chars=int(getattr(args, "prompt_chunk_chars", 400)),
        system_prompt_mode=system_prompt,
        few_shot=bool(getattr(args, "few_shot", False))
    )
    prompt_ms = (time.perf_counter() - p0) * 1000.0

    # Abstain if low confidence (only when using retrieval)
    abstain_threshold = float(getattr(args, "abstain_threshold", 0.2))
    try:
        if not (golden_chunks and cfg.use_golden_chunks) and not cfg.disable_chunks:
            if confidence < abstain_threshold:
                ans = "I don't know"
                generation_ms = 0.0
                tokens_generated = 0
                stats = {
                    "retrieval_ms": retrieval_ms,
                    "ranking_ms": ranking_ms,
                    "prompt_ms": prompt_ms,
                    "generation_ms": generation_ms,
                    "model_load_ms": 0.0,
                    "tokens_generated": float(tokens_generated),
                    "tokens_per_sec": 0.0,
                    "confidence": float(confidence),
                }
                return ans, stats, context_items
    except Exception:
        pass

    # Step 4: Generation
    pre_llm = get_llm_stats()
    if getattr(args, "sources_only", False):
        ans = ""
        generation_ms = 0.0
    else:
        g0 = time.perf_counter()
        ans = answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
            prompt_chunk_chars=int(getattr(args, "prompt_chunk_chars", 400)),
            few_shot=bool(getattr(args, "few_shot", False)),
            **({"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0} if getattr(args, "deterministic", False) else {})
        )
    generation_ms = (time.perf_counter() - g0) * 1000.0
    post_llm = get_llm_stats()
    model_load_ms = 0.0
    try:
        if post_llm.get("misses", 0) > pre_llm.get("misses", 0):
            model_load_ms = float(post_llm.get("last_load_ms") or 0.0)
    except Exception:
        model_load_ms = 0.0

    # Rough token count estimate if backend doesn't provide usage
    tokens_generated = max(1, len(ans) // 4)

    stats = {
        "retrieval_ms": retrieval_ms,
        "ranking_ms": ranking_ms,
        "prompt_ms": prompt_ms,
        "generation_ms": generation_ms,
        "model_load_ms": model_load_ms,
        "tokens_generated": float(tokens_generated),
        "tokens_per_sec": (tokens_generated / (generation_ms / 1000.0)) if generation_ms > 0 else 0.0,
        "confidence": float(confidence),
    }

    return ans, stats, context_items


def run_chat_session(args: argparse.Namespace, cfg: QueryPlanConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()
    # planner = HeuristicQueryPlanner(cfg)

    # Load artifacts, initialize retrievers and rankers once before the loop.
    print("Welcome to Tokensmith! Initializing chat...")
    try:
        # Disabled till we fix the core pipeline
        # cfg = planner.plan(q)
        artifacts_dir = cfg.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources = load_artifacts(
            artifacts_dir=artifacts_dir, 
            index_prefix=args.index_prefix
        )

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )
        
        # Package artifacts for reuse
        artifacts = {
            "chunks": chunks,
            "sources": sources,
            "retrievers": retrievers,
            "ranker": ranker
        }
    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            # Timing: start
            start_wall = datetime.now().isoformat(timespec='seconds')
            start_t = time.perf_counter()
            print(f"[TIMING] start={start_wall}")

            # Respect --no_cache_model setting
            set_model_cache_enabled(not args.no_cache_model)

            # Use the single query function
            ans, stats, context_items = get_answer(q, cfg, args, logger=logger,artifacts=artifacts)

            if not args.sources_only:
                print("\n=================== START OF ANSWER ===================")
                print(ans.strip() if ans and ans.strip() else "(No output from model)")
                print("\n==================== END OF ANSWER ====================")
                logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path or cfg.model_path})
            # Optionally print retrieved context
            if args.show_context or args.sources_only:
                print("\n=================== RETRIEVED CONTEXT ===================")
                for item in context_items:
                    sf = float(item.get('score_faiss', 0.0))
                    sb_raw = float(item.get('score_bm25_raw', 0.0))
                    sb_sat = float(item.get('score_bm25_sat', 0.0))
                    sfused = float(item.get('score_fused', 0.0))
                    sfused_w = float(item.get('score_fused_warped', 0.0))
                    print(f"[S{item['S']}] {item['source']} (idx={item['index']}) | fused_warped={sfused_w:.3f} fused={sfused:.3f} faiss={sf:.3f} bm25_sat={sb_sat:.3f} bm25_raw={sb_raw:.2f}\n{item['preview']}")
                    print("------------------------------------------------------")
                print("================= END RETRIEVED CONTEXT ================")
            # Stage-level timing
            print(f"[TIMING] retrieval_ms={stats['retrieval_ms']:.1f} ranking_ms={stats['ranking_ms']:.1f} prompt_ms={stats['prompt_ms']:.1f} model_load_ms={stats['model_load_ms']:.1f} generation_ms={stats['generation_ms']:.1f} tokens_generated={int(stats['tokens_generated'])} tokens_per_sec={stats['tokens_per_sec']:.2f} confidence={stats.get('confidence', 0.0):.2f}")
            

            # Timing: end
            end_wall = datetime.now().isoformat(timespec='seconds')
            elapsed = time.perf_counter() - start_t
            print(f"[TIMING] end={end_wall} elapsed={elapsed:.2f}s")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.log_error(str(e))
            break

    # TODO: Fix completion logging.
    # logger.log_query_complete()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Config loading
    config_path = pathlib.Path("config/config.yaml")
    cfg = None
    if config_path.exists():
        cfg = QueryPlanConfig.from_yaml(config_path)

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided and no fallback found at config/ or ~/.config/tokensmith/"
        )

    init_logger(cfg)

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
