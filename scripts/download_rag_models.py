import argparse
import os
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from health_guide.config import RAG_EMBED_MODEL_NAME, RAG_RERANK_MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Download RAG embedding and reranker models for offline use."
    )
    parser.add_argument(
        "--embed-model",
        default=RAG_EMBED_MODEL_NAME,
        help="Embedding model name or local path",
    )
    parser.add_argument(
        "--rerank-model",
        default=RAG_RERANK_MODEL_NAME,
        help="Reranker model name or local path",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional Hugging Face cache dir (e.g. ./.hf_cache)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Download/load device for warmup: cpu or cuda",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint, default uses hf-mirror for users in mainland China",
    )
    parser.add_argument(
        "--disable-mirror",
        action="store_true",
        help="Disable mirror and use default huggingface endpoint",
    )
    args = parser.parse_args()

    if not args.disable_mirror:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        # 兼容部分库读取的备用变量
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = args.hf_endpoint
        print(f"[Model Download] Using mirror endpoint: {args.hf_endpoint}")
    else:
        print("[Model Download] Mirror disabled. Using default Hugging Face endpoint.")

    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).resolve())
        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Model Download] Cache dir: {cache_dir}")

    sentence_transformers = importlib.import_module("sentence_transformers")
    CrossEncoder = sentence_transformers.CrossEncoder
    SentenceTransformer = sentence_transformers.SentenceTransformer

    print(f"[Model Download] Embedding model: {args.embed_model}")
    embed_model = SentenceTransformer(args.embed_model, device=args.device)
    _ = embed_model.encode(["模型下载检查"], show_progress_bar=False)

    print(f"[Model Download] Reranker model: {args.rerank_model}")
    reranker = CrossEncoder(args.rerank_model, device=args.device, max_length=256)
    _ = reranker.predict([["query", "passage"]], show_progress_bar=False)

    print("[Model Download] Done. Models are cached and ready for offline runs.")


if __name__ == "__main__":
    main()
