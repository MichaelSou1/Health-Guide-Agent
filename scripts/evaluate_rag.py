import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EvalSample:
    query: str
    agent: str
    relevant_sources: Set[str]
    relevant_chunk_ids: Set[str]


def load_dataset(path: Path) -> List[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: List[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = (row.get("query") or "").strip()
            if not query:
                raise ValueError(f"Line {line_no}: missing non-empty query")
            agent = (row.get("agent") or "general").strip().lower()

            sources = set(row.get("relevant_sources") or [])
            chunks = set(row.get("relevant_chunk_ids") or [])
            if not sources and not chunks:
                raise ValueError(
                    f"Line {line_no}: provide at least one of relevant_sources/relevant_chunk_ids"
                )

            samples.append(
                EvalSample(
                    query=query,
                    agent=agent,
                    relevant_sources=sources,
                    relevant_chunk_ids=chunks,
                )
            )
    return samples


def _is_relevant(item: Dict[str, str], sample: EvalSample) -> bool:
    source = item.get("source") or ""
    chunk_id = item.get("chunk_id") or ""
    source_match = bool(
        sample.relevant_sources
        and any(source == s or source.endswith("/" + s) for s in sample.relevant_sources)
    )
    chunk_match = bool(
        sample.relevant_chunk_ids
        and any(chunk_id == c or chunk_id.endswith(c) for c in sample.relevant_chunk_ids)
    )
    return source_match or chunk_match


def _first_relevant_rank(results: List[Dict[str, str]], sample: EvalSample) -> Optional[int]:
    for idx, r in enumerate(results, start=1):
        if _is_relevant(r, sample):
            return idx
    return None


def _recall_at_k(results: List[Dict[str, str]], sample: EvalSample, k: int) -> float:
    top = results[:k]

    if sample.relevant_chunk_ids:
        gt = sample.relevant_chunk_ids
        hit = {r.get("chunk_id") for r in top if r.get("chunk_id") in gt}
        return len(hit) / len(gt)

    # source-level recall
    gt = sample.relevant_sources
    hit_count = 0
    for g in gt:
        if any((r.get("source") or "") == g or (r.get("source") or "").endswith("/" + g) for r in top):
            hit_count += 1
    return hit_count / len(gt)


def _hit_at_k(results: List[Dict[str, str]], sample: EvalSample, k: int) -> int:
    top = results[:k]
    return 1 if any(_is_relevant(r, sample) for r in top) else 0


def evaluate(samples: List[EvalSample], kb: "LayeredKnowledgeRouter", ks: List[int]) -> Dict[str, object]:
    max_k = max(ks)

    recall_sums = {k: 0.0 for k in ks}
    hit_sums = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0

    details = []
    per_agent_buckets: Dict[str, List[Dict[str, float]]] = {}

    for sample in samples:
        results = kb.retrieve(sample.query, agent=sample.agent, top_k=max_k)
        first_rank = _first_relevant_rank(results, sample)
        rr = 1.0 / first_rank if first_rank else 0.0
        reciprocal_rank_sum += rr

        row = {
            "query": sample.query,
            "agent": sample.agent,
            "first_relevant_rank": first_rank,
            "rr": rr,
            "top_results": [
                {
                    "rank": i + 1,
                    "source": r.get("source"),
                    "chunk_id": r.get("chunk_id"),
                    "score": r.get("score"),
                }
                for i, r in enumerate(results)
            ],
        }

        for k in ks:
            r_k = _recall_at_k(results, sample, k)
            h_k = _hit_at_k(results, sample, k)
            recall_sums[k] += r_k
            hit_sums[k] += h_k
            row[f"recall@{k}"] = r_k
            row[f"hit@{k}"] = h_k

        details.append(row)

        if sample.agent not in per_agent_buckets:
            per_agent_buckets[sample.agent] = []
        per_agent_buckets[sample.agent].append(
            {
                "rr": rr,
                **{f"recall@{k}": row[f"recall@{k}"] for k in ks},
                **{f"hit@{k}": row[f"hit@{k}"] for k in ks},
            }
        )

    n = len(samples)
    summary = {
        "sample_count": n,
        "mrr": round(reciprocal_rank_sum / n, 4) if n else 0.0,
        "recall": {f"recall@{k}": round(recall_sums[k] / n, 4) if n else 0.0 for k in ks},
        "hit_rate": {f"hit_rate@{k}": round(hit_sums[k] / n, 4) if n else 0.0 for k in ks},
    }

    per_agent_summary = {}
    for agent, rows in per_agent_buckets.items():
        m = len(rows)
        if m == 0:
            continue

        per_agent_summary[agent] = {
            "sample_count": m,
            "mrr": round(sum(x["rr"] for x in rows) / m, 4),
            "recall": {
                f"recall@{k}": round(sum(x[f"recall@{k}"] for x in rows) / m, 4)
                for k in ks
            },
            "hit_rate": {
                f"hit_rate@{k}": round(sum(x[f"hit@{k}"] for x in rows) / m, 4)
                for k in ks
            },
        }

    return {
        "summary": summary,
        "per_agent_summary": per_agent_summary,
        "details": details,
    }


def parse_ks(raw: str) -> List[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    ks = sorted({int(x) for x in items})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--ks must contain positive integers, e.g. 1,3,5")
    return ks


def main():
    from health_guide.config import KNOWLEDGE_BASE_DIR
    from health_guide.rag import LayeredKnowledgeRouter

    parser = argparse.ArgumentParser(description="Evaluate local RAG with Recall@k / MRR / Hit Rate")
    parser.add_argument(
        "--dataset",
        default="eval/rag_eval_dataset.jsonl",
        help="Path to JSONL eval dataset",
    )
    parser.add_argument("--kb-dir", default=KNOWLEDGE_BASE_DIR, help="Knowledge base directory")
    parser.add_argument("--chunk-size", type=int, default=420, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap")
    parser.add_argument("--ks", default="1,3,5", help="Comma-separated k list, e.g. 1,3,5")
    parser.add_argument(
        "--out",
        default="reports/rag_eval_report.json",
        help="Path to output evaluation report JSON",
    )
    args = parser.parse_args()

    ks = parse_ks(args.ks)
    dataset = load_dataset(Path(args.dataset))

    kb = LayeredKnowledgeRouter(kb_root=args.kb_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    kb.build(force_rebuild=False)

    report = evaluate(dataset, kb, ks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[RAG Eval] Completed")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[RAG Eval] Report written to: {out}")


if __name__ == "__main__":
    main()
