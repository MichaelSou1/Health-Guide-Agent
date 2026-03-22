import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from health_guide.config import KNOWLEDGE_BASE_DIR
    from health_guide.rag import LayeredKnowledgeRouter

    parser = argparse.ArgumentParser(description="Offline prebuild for local RAG embeddings/index cache.")
    parser.add_argument("--kb-dir", default=KNOWLEDGE_BASE_DIR, help="Knowledge base directory path")
    parser.add_argument("--chunk-size", type=int, default=420, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap")
    parser.add_argument(
        "--agent",
        default="",
        help="Optional agent namespace to prebuild (trainer/nutritionist/wellness/general). Empty means build all.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild: clear cache and regenerate")
    parser.add_argument(
        "--stats-out",
        default="reports/rag_index_stats.json",
        help="Output path for index stats JSON",
    )
    args = parser.parse_args()

    kb = LayeredKnowledgeRouter(kb_root=args.kb_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    kb.build(force_rebuild=args.rebuild, agent=args.agent or None)

    stats = kb.get_index_stats()
    payload = {
        "layered_stats": stats,
        "kb_dir": str(Path(args.kb_dir)),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "agent": args.agent or "all",
        "force_rebuild": args.rebuild,
    }

    out = Path(args.stats_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[RAG Index] Build complete")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[RAG Index] Stats written to: {out}")


if __name__ == "__main__":
    main()
