from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import os

# Configure writable caches for read-only environments (e.g., Vercel). Must run before importing ML libs.
CACHE_ROOT = os.environ.get("EMBED_CACHE_ROOT", "/tmp")
if (
    os.environ.get("VERCEL") == "1"
    or os.environ.get("READ_ONLY_FS") == "1"
    or not os.access("/", os.W_OK)
):
    hf_home = os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(CACHE_ROOT, "cache"))
    # Best-effort create dirs; ignore failures in restricted envs
    for key in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
    ):
        try:
            os.makedirs(os.environ[key], exist_ok=True)
        except Exception:
            pass

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagAutoReranker


app = FastAPI(title="Embedding Service", version="0.1.0")


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


# Initialize the embedding model at startup (lazy fallback if initial init fails)
try:
    embedding_model = SentenceTransformer(MODEL_NAME)
    model_init_error = None
except Exception as exc:  # pragma: no cover - environment dependent
    embedding_model = None
    model_init_error = exc


# Initialize the reranker model at startup (lazy fallback if initial init fails)
try:
    reranker_model = FlagAutoReranker.from_finetuned(RERANKER_MODEL_NAME)
    reranker_init_error = None
except Exception as exc:  # pragma: no cover - environment dependent
    reranker_model = None
    reranker_init_error = exc


class EmbedRequest(BaseModel):
    text: str
    role: Literal["query", "passage"]


class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dim: int


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


class RerankResponse(BaseModel):
    scores: List[float]
    model: str


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ready": embedding_model is not None and model_init_error is None,
        "reranker_model": RERANKER_MODEL_NAME,
        "reranker_ready": reranker_model is not None and reranker_init_error is None,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        global embedding_model, model_init_error

        # Lazy re-init if startup failed
        if embedding_model is None:
            embedding_model = SentenceTransformer(MODEL_NAME)
            model_init_error = None

        prefixed = f"{req.role}: {req.text}"
        vectors = embedding_model.encode([prefixed], normalize_embeddings=True)
        if len(vectors) == 0:
            raise RuntimeError("No embedding returned from model")

        vec = vectors[0]
        embedding_list = [float(x) for x in vec]

        return EmbedResponse(
            embedding=embedding_list,
            model=MODEL_NAME,
            dim=len(embedding_list),
        )
    except Exception as e:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    try:
        global reranker_model, reranker_init_error

        # Lazy re-init if startup failed
        if reranker_model is None:
            reranker_model = FlagAutoReranker.from_finetuned(RERANKER_MODEL_NAME)
            reranker_init_error = None

        if not req.documents:
            return RerankResponse(scores=[], model=RERANKER_MODEL_NAME)

        pairs = [(req.query, doc) for doc in req.documents]
        scores = reranker_model.compute_score(pairs, normalize=True)

        score_list = [float(s) for s in scores]

        return RerankResponse(scores=score_list, model=RERANKER_MODEL_NAME)
    except Exception as e:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=f"Rerank error: {e}")


