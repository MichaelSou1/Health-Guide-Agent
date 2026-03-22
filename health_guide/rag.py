import re
import json
import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from .config import (
    KNOWLEDGE_BASE_DIR,
    KNOWLEDGE_BASE_SHARED_SUBDIR,
    KNOWLEDGE_BASE_AGENT_SUBDIRS,
    RAG_EMBED_BATCH_SIZE,
    RAG_EMBED_MODEL_NAME,
    RAG_FINAL_TOP_K,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_MODEL_NAME,
    RAG_RETRIEVE_TOP_K,
    RAG_DEVICE,
)


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


class LocalKnowledgeBase:
    """本地 RAG：两阶段检索（Dense Retrieve + Cross-Encoder Re-rank）。"""

    def __init__(
        self,
        kb_dir: str = KNOWLEDGE_BASE_DIR,
        chunk_size: int = 420,
        overlap: int = 80,
        recursive: bool = True,
    ):
        self.kb_dir = Path(kb_dir)
        self.cache_dir = self.kb_dir / ".index_cache"
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.recursive = recursive
        self.chunks: List[Chunk] = []

        # 运行时对象（延迟加载，减少冷启动）
        self._np = None
        self._embed_model = None
        self._rerank_model = None
        self._device = None

        # 索引缓存
        self._chunk_embeddings = None
        self._fingerprint = None
        self._ready = False

    def _read_documents(self) -> List[Dict[str, str]]:
        if not self.kb_dir.exists():
            return []

        docs = []
        iter_files = self.kb_dir.rglob("*") if self.recursive else self.kb_dir.glob("*")
        for path in sorted(iter_files):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                docs.append({"source": str(path.relative_to(self.kb_dir)).replace("\\", "/"), "text": content})
        return docs

    def _docs_fingerprint(self, docs: List[Dict[str, str]]) -> str:
        payload = {
            "docs": docs,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embed_model": RAG_EMBED_MODEL_NAME,
        }
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _resolve_device(self) -> str:
        if RAG_DEVICE != "auto":
            return RAG_DEVICE

        try:
            torch = importlib.import_module("torch")
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _lazy_import_numpy(self):
        if self._np is None:
            self._np = importlib.import_module("numpy")
        return self._np

    def _lazy_load_models(self):
        if self._embed_model is not None and self._rerank_model is not None:
            return

        sentence_transformers = importlib.import_module("sentence_transformers")
        torch = importlib.import_module("torch")

        self._device = self._resolve_device()
        use_fp16 = self._device == "cuda"

        self._embed_model = sentence_transformers.SentenceTransformer(
            RAG_EMBED_MODEL_NAME,
            device=self._device,
        )

        self._rerank_model = sentence_transformers.CrossEncoder(
            RAG_RERANK_MODEL_NAME,
            device=self._device,
            max_length=256,
        )

        # 端侧优化：GPU 场景启用半精度，减少显存和延迟
        if use_fp16:
            try:
                self._embed_model.half()
                self._rerank_model.model.half()
            except Exception:
                pass

        # 端侧优化：开启推理优化（若可用）
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _cache_meta_path(self) -> Path:
        return self.cache_dir / "index_meta.json"

    def _cache_embeddings_path(self) -> Path:
        return self.cache_dir / "embeddings.npy"

    def _cache_chunks_path(self) -> Path:
        return self.cache_dir / "chunks.json"

    def _try_load_cache(self) -> bool:
        np = self._lazy_import_numpy()
        meta_path = self._cache_meta_path()
        emb_path = self._cache_embeddings_path()
        chunks_path = self._cache_chunks_path()
        if not (meta_path.exists() and emb_path.exists() and chunks_path.exists()):
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != self._fingerprint:
                return False

            chunk_items = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.chunks = [Chunk(**item) for item in chunk_items]
            self._chunk_embeddings = np.load(str(emb_path))
            return True
        except Exception:
            return False

    def _save_cache(self):
        np = self._lazy_import_numpy()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_meta_path().write_text(
            json.dumps({"fingerprint": self._fingerprint}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._cache_chunks_path().write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(str(self._cache_embeddings_path()), self._chunk_embeddings)

    def _split_text(self, text: str) -> List[str]:
        clean = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not clean:
            return []

        chunks = []
        start = 0
        n = len(clean)
        while start < n:
            end = min(start + self.chunk_size, n)
            piece = clean[start:end]
            chunks.append(piece)
            if end == n:
                break
            start = max(0, end - self.overlap)
        return chunks

    def clear_cache(self):
        if self.cache_dir.exists():
            for p in self.cache_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(self.cache_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()

        self._chunk_embeddings = None
        self.chunks = []
        self._ready = False

    def build(self, force_rebuild: bool = False):
        if force_rebuild:
            self.clear_cache()

        docs = self._read_documents()
        self._fingerprint = self._docs_fingerprint(docs)

        chunks: List[Chunk] = []
        for d in docs:
            pieces = self._split_text(d["text"])
            for i, p in enumerate(pieces):
                chunks.append(
                    Chunk(
                        chunk_id=f"{d['source']}#chunk-{i+1}",
                        source=d["source"],
                        text=p,
                    )
                )

        self.chunks = chunks

        if not self.chunks:
            self._chunk_embeddings = None
            self._ready = True
            return

        # 优先加载缓存索引，加速冷启动
        if self._try_load_cache():
            self._ready = True
            return

        self._lazy_load_models()
        np = self._lazy_import_numpy()

        texts = [c.text for c in self.chunks]
        embeddings = self._embed_model.encode(
            texts,
            batch_size=RAG_EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._chunk_embeddings = np.asarray(embeddings, dtype=np.float32)
        self._save_cache()
        self._ready = True

    def get_index_stats(self) -> Dict[str, int]:
        return {
            "doc_count": len(self._read_documents()),
            "chunk_count": len(self.chunks),
            "cache_exists": int(self.cache_dir.exists()),
        }

    def retrieve(self, query: str, top_k: int = RAG_FINAL_TOP_K) -> List[Dict[str, str]]:
        if not self._ready:
            self.build()

        if not self.chunks or self._chunk_embeddings is None:
            return []

        query = (query or "").strip()
        if not query:
            return []

        self._lazy_load_models()
        np = self._lazy_import_numpy()

        # Stage-1: Dense Retrieve
        query_emb = self._embed_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        sim_scores = self._chunk_embeddings @ query_emb
        retrieve_k = min(max(top_k, RAG_RETRIEVE_TOP_K), len(self.chunks))
        candidate_idx = np.argpartition(-sim_scores, retrieve_k - 1)[:retrieve_k]
        candidate_idx = sorted(candidate_idx.tolist(), key=lambda i: float(sim_scores[i]), reverse=True)

        # Stage-2: Re-rank
        pairs = [[query, self.chunks[i].text] for i in candidate_idx]
        rerank_scores = self._rerank_model.predict(
            pairs,
            batch_size=RAG_RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

        combined = []
        for i, rr in zip(candidate_idx, rerank_scores):
            dense = float(sim_scores[i])
            rerank = float(rr)
            # 组合分数（以重排分为主，保留召回分作微调）
            final_score = rerank + 0.15 * dense
            combined.append((i, dense, rerank, final_score))

        combined.sort(key=lambda x: x[3], reverse=True)
        final_k = min(max(1, top_k), len(combined))
        selected = combined[:final_k]

        results = []
        for idx, dense_score, rerank_score, final_score in selected:
            c = self.chunks[idx]
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "score": round(final_score, 4),
                    "dense_score": round(dense_score, 4),
                    "rerank_score": round(rerank_score, 4),
                    "content": c.text,
                }
            )
        return results

class LayeredKnowledgeRouter:
    """分层知识路由：共享知识库 + Agent 私有知识库。"""

    def __init__(self, kb_root: str = KNOWLEDGE_BASE_DIR, chunk_size: int = 420, overlap: int = 80):
        self.kb_root = Path(kb_root)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self._shared_store = self._build_shared_store()
        self._agent_stores = self._build_agent_stores()
        self._ready = False

    def _build_shared_store(self) -> LocalKnowledgeBase:
        shared_path = self.kb_root / KNOWLEDGE_BASE_SHARED_SUBDIR

        # 若 shared 子目录存在，优先使用；否则兼容旧结构：直接读取根目录下的顶层文档。
        if shared_path.exists():
            return LocalKnowledgeBase(
                kb_dir=str(shared_path),
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                recursive=True,
            )

        return LocalKnowledgeBase(
            kb_dir=str(self.kb_root),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            recursive=False,
        )

    def _build_agent_stores(self) -> Dict[str, LocalKnowledgeBase]:
        stores = {}
        for agent, subdir in KNOWLEDGE_BASE_AGENT_SUBDIRS.items():
            agent_path = self.kb_root / subdir
            stores[agent] = LocalKnowledgeBase(
                kb_dir=str(agent_path),
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                recursive=True,
            )
        return stores

    def build(self, force_rebuild: bool = False, agent: Optional[str] = None):
        self._shared_store.build(force_rebuild=force_rebuild)

        if agent:
            target = self._agent_stores.get(agent)
            if target:
                target.build(force_rebuild=force_rebuild)
        else:
            for store in self._agent_stores.values():
                store.build(force_rebuild=force_rebuild)

        self._ready = True

    def _retrieve_agent_private(self, query: str, agent: str, top_k: int) -> List[Dict[str, str]]:
        store = self._agent_stores.get(agent)
        if not store:
            return []
        return store.retrieve(query=query, top_k=max(top_k, RAG_RETRIEVE_TOP_K // 2))

    def _retrieve_shared(self, query: str, top_k: int) -> List[Dict[str, str]]:
        return self._shared_store.retrieve(query=query, top_k=max(top_k, RAG_RETRIEVE_TOP_K // 2))

    @staticmethod
    def _decorate_results(items: List[Dict[str, str]], namespace: str, boost: float) -> List[Dict[str, str]]:
        enriched = []
        for item in items:
            x = dict(item)
            base_score = float(x.get("score", 0.0))
            x["namespace"] = namespace
            x["chunk_id"] = f"{namespace}:{x.get('chunk_id', '')}"
            x["source"] = f"{namespace}/{x.get('source', '')}"
            x["score"] = round(base_score + boost, 4)
            enriched.append(x)
        return enriched

    @staticmethod
    def _dedupe(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        merged = []
        for r in results:
            key = r.get("chunk_id")
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        return merged

    def retrieve(self, query: str, agent: str = "general", top_k: int = RAG_FINAL_TOP_K) -> List[Dict[str, str]]:
        if not self._ready:
            self.build(force_rebuild=False, agent=agent)

        private_results = self._retrieve_agent_private(query=query, agent=agent, top_k=top_k)
        shared_results = self._retrieve_shared(query=query, top_k=top_k)

        # 分层融合：私有知识优先（轻量打分增益），共享知识补充。
        merged = []
        merged.extend(self._decorate_results(private_results, namespace=agent, boost=0.2))
        merged.extend(self._decorate_results(shared_results, namespace="shared", boost=0.0))
        merged = self._dedupe(merged)
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        return merged[: max(1, top_k)]

    def get_index_stats(self) -> Dict[str, Dict[str, int]]:
        stats = {"shared": self._shared_store.get_index_stats()}
        for agent, store in self._agent_stores.items():
            stats[agent] = store.get_index_stats()
        return stats


_router_singleton = LayeredKnowledgeRouter()


def get_kb() -> LayeredKnowledgeRouter:
    return _router_singleton


def get_router() -> LayeredKnowledgeRouter:
    return _router_singleton
