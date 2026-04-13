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


# 目前支持的知识语料文件类型。
# - .md / .txt: 直接按 UTF-8 读取
# - .pdf: 通过 pypdf 按页提取文本,用 \f (form-feed) 作为分页分隔符,
#         便于后续 chunk 归页
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
PAGE_SEPARATOR = "\f"


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    page_range: Optional[str] = None  # 仅 PDF 有值, 例如 "3" 或 "3-4"


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
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue

            source = str(path.relative_to(self.kb_dir)).replace("\\", "/")

            if suffix == ".pdf":
                content = self._read_pdf(path)
                file_type = "pdf"
            else:
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                file_type = "text"

            if not content or not content.strip():
                # 扫描版 PDF / 空文件等情况:跳过,避免污染索引
                continue

            docs.append({"source": source, "text": content, "file_type": file_type})
        return docs

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """按页提取 PDF 文本,用 form-feed (\\f) 作为页分隔符。

        设计说明:
        - 用 pypdf 进行纯 Python 解析,无需系统依赖,部署方便。
        - 使用 \\f 作为分页符,后续 chunker 可以通过计算 \\f 出现次数推算 chunk 所在页范围,
          而又不会因为分页符干扰 embedding 文本语义(\\f 不携带语义,BGE 类 tokenizer 会直接忽略)。
        - 对单页文本做轻量清洗:合并重复换行、去除行尾空白,减少 PDF 提取时的噪声。
        - 扫描版 PDF 会得到空字符串 —— 由调用方负责跳过;OCR 不在本项目范围内。
        """
        try:
            pypdf = importlib.import_module("pypdf")
        except ImportError as e:
            raise ImportError(
                "解析 PDF 需要 pypdf,请先执行: pip install pypdf"
            ) from e

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as e:
            print(f"[RAG][warn] 无法解析 PDF: {path} ({e.__class__.__name__}: {e})")
            return ""

        pages_text: List[str] = []
        for page in reader.pages:
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            # 轻量清洗:压缩重复空行,去掉行尾空白
            cleaned = re.sub(r"[ \t]+\n", "\n", raw)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            # 页尾补一个换行,防止后续 chunk 跨页时两页文字被直接拼接成无意义长词
            if cleaned:
                cleaned += "\n"
            pages_text.append(cleaned)

        return PAGE_SEPARATOR.join(pages_text)

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

    def _split_text(self, text: str):
        """切分文本为 chunks。返回 [(piece, page_range_or_None), ...]。

        如果 text 中包含 PAGE_SEPARATOR(来自 PDF 提取),会建立「char offset → 页码」映射,
        然后在切分后反推每个 chunk 所跨的页码区间,用于 citation。
        普通 md/txt 文件则始终返回 page_range=None。
        """
        has_pages = PAGE_SEPARATOR in text
        page_numbers: Optional[List[int]] = None

        if has_pages:
            cleaned_chars: List[str] = []
            page_numbers = []
            current_page = 1
            for ch in text:
                if ch == PAGE_SEPARATOR:
                    current_page += 1
                    continue
                cleaned_chars.append(ch)
                page_numbers.append(current_page)
            cleaned = "".join(cleaned_chars)
        else:
            cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not cleaned.strip():
            return []

        results: List = []
        start = 0
        n = len(cleaned)
        while start < n:
            end = min(start + self.chunk_size, n)
            piece = cleaned[start:end].strip()

            if piece:
                if page_numbers is not None:
                    start_page = page_numbers[start]
                    end_page = page_numbers[end - 1]
                    page_range = (
                        str(start_page)
                        if start_page == end_page
                        else f"{start_page}-{end_page}"
                    )
                else:
                    page_range = None
                results.append((piece, page_range))

            if end == n:
                break
            start = max(start + 1, end - self.overlap)
        return results

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
            for i, (piece_text, page_range) in enumerate(pieces):
                if page_range:
                    chunk_id = f"{d['source']}#p{page_range}-chunk-{i+1}"
                else:
                    chunk_id = f"{d['source']}#chunk-{i+1}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=d["source"],
                        text=piece_text,
                        page_range=page_range,
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

    def retrieve_stages(
        self,
        query: str,
        stage1_k: int = RAG_RETRIEVE_TOP_K,
        stage2_k: int = RAG_FINAL_TOP_K,
    ) -> Dict[str, List[Dict[str, object]]]:
        """返回两阶段检索的原始结果，用于对 Embedding 召回与 Rerank 质量分别做评测。

        - stage1: 仅使用 embedding 的稠密检索 Top-K（按余弦相似度排序）
        - stage2: 对 stage1 的候选做 cross-encoder 重排后的 Top-K（按 rerank 分排序）

        与 `retrieve()` 不同，这里不做任何分数融合或 boost，保持原始分数以便公平对比两个阶段。
        """
        empty: Dict[str, List[Dict[str, object]]] = {"stage1": [], "stage2": []}
        if not self._ready:
            self.build()

        if not self.chunks or self._chunk_embeddings is None:
            return empty

        query = (query or "").strip()
        if not query:
            return empty

        self._lazy_load_models()
        np = self._lazy_import_numpy()

        stage1_k = max(1, min(stage1_k, len(self.chunks)))
        stage2_k = max(1, min(stage2_k, stage1_k))

        # Stage-1: Dense Retrieve
        query_emb = self._embed_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        sim_scores = self._chunk_embeddings @ query_emb
        candidate_idx = np.argpartition(-sim_scores, stage1_k - 1)[:stage1_k]
        candidate_idx = sorted(
            candidate_idx.tolist(), key=lambda i: float(sim_scores[i]), reverse=True
        )

        stage1_results: List[Dict[str, object]] = []
        for rank, i in enumerate(candidate_idx, start=1):
            c = self.chunks[i]
            stage1_results.append(
                {
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page_range": c.page_range,
                    "dense_score": round(float(sim_scores[i]), 4),
                }
            )

        # Stage-2: Re-rank
        pairs = [[query, self.chunks[i].text] for i in candidate_idx]
        rerank_scores = self._rerank_model.predict(
            pairs,
            batch_size=RAG_RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

        combined = []
        for i, rr in zip(candidate_idx, rerank_scores):
            combined.append((i, float(sim_scores[i]), float(rr)))
        combined.sort(key=lambda x: x[2], reverse=True)

        stage2_results: List[Dict[str, object]] = []
        for rank, (i, dense, rr) in enumerate(combined[:stage2_k], start=1):
            c = self.chunks[i]
            stage2_results.append(
                {
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page_range": c.page_range,
                    "dense_score": round(dense, 4),
                    "rerank_score": round(rr, 4),
                }
            )

        return {"stage1": stage1_results, "stage2": stage2_results}

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
                    "page_range": c.page_range,
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

    def retrieve_stages(
        self,
        query: str,
        agent: str = "general",
        stage1_k: int = RAG_RETRIEVE_TOP_K,
        stage2_k: int = RAG_FINAL_TOP_K,
    ) -> Dict[str, List[Dict[str, object]]]:
        """分层路由版的两阶段检索结果（用于评测）。

        合并 agent 私有库 + 共享库的候选，stage1 按 dense_score 排序、stage2 按 rerank_score 排序。
        为了便于评测指标与生产行为对齐，chunk_id 和 source 会带上 namespace 前缀。
        不做额外的 score boost——评测时只关心召回/排序本身，而不是融合策略。
        """
        if not self._ready:
            self.build(force_rebuild=False, agent=agent)

        per_store_stage1_k = max(stage1_k, RAG_RETRIEVE_TOP_K // 2)
        per_store_stage2_k = max(stage2_k, RAG_FINAL_TOP_K)

        parts_stage1: List[Dict[str, object]] = []
        parts_stage2: List[Dict[str, object]] = []

        private_store = self._agent_stores.get(agent)
        if private_store is not None:
            p = private_store.retrieve_stages(
                query=query, stage1_k=per_store_stage1_k, stage2_k=per_store_stage2_k
            )
            parts_stage1.extend(self._tag_namespace(p["stage1"], namespace=agent))
            parts_stage2.extend(self._tag_namespace(p["stage2"], namespace=agent))

        s = self._shared_store.retrieve_stages(
            query=query, stage1_k=per_store_stage1_k, stage2_k=per_store_stage2_k
        )
        parts_stage1.extend(self._tag_namespace(s["stage1"], namespace="shared"))
        parts_stage2.extend(self._tag_namespace(s["stage2"], namespace="shared"))

        parts_stage1 = self._dedupe(parts_stage1)
        parts_stage2 = self._dedupe(parts_stage2)

        parts_stage1.sort(key=lambda x: float(x.get("dense_score", 0.0)), reverse=True)
        parts_stage2.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)

        # 重新编号
        for rank, r in enumerate(parts_stage1[:stage1_k], start=1):
            r["rank"] = rank
        for rank, r in enumerate(parts_stage2[:stage2_k], start=1):
            r["rank"] = rank

        return {
            "stage1": parts_stage1[:stage1_k],
            "stage2": parts_stage2[:stage2_k],
        }

    @staticmethod
    def _tag_namespace(items: List[Dict[str, object]], namespace: str) -> List[Dict[str, object]]:
        tagged = []
        for item in items:
            x = dict(item)
            x["namespace"] = namespace
            x["chunk_id"] = f"{namespace}:{x.get('chunk_id', '')}"
            x["source"] = f"{namespace}/{x.get('source', '')}"
            tagged.append(x)
        return tagged

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
