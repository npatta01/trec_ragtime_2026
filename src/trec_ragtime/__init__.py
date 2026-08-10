"""Reusable client for the TREC RAGTIME retrieval service."""

from .cache import SQLiteResponseCache, default_cache_path
from .client import (
    DEFAULT_CONTENT_COLLECTION,
    DEFAULT_PIPELINE,
    RagtimeApiError,
    RagtimeClient,
    RagtimeConfigurationError,
    RagtimeError,
    RagtimeProtocolError,
)
from .huggingface import (
    DEFAULT_HF_BASE_URL,
    DEFAULT_HF_CONFIG,
    DEFAULT_HF_DATASET,
    DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS,
    DEFAULT_HF_MT_CONFIG,
    DEFAULT_HF_MT_SPLITS,
    DEFAULT_HF_SPLITS,
    HuggingFaceApiError,
    HuggingFaceConfigurationError,
    HuggingFaceCorpus,
    HuggingFaceError,
    HuggingFaceHydrationError,
    HuggingFaceProtocolError,
)
from .local_corpus import (
    LocalCorpusBuildError,
    LocalCorpusConfigurationError,
    LocalCorpusError,
    LocalCorpusIndex,
    LocalCorpusMetadata,
    LocalCorpusReadError,
    LocalCorpusValidationError,
    build_from_parquet,
    default_local_corpus_path,
)
from .models import (
    Document,
    HuggingFaceDocument,
    HydrationFailure,
    HydrationResult,
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from .passages import (
    Boundary,
    Passage,
    PassageChunker,
    PassageChunkingError,
    Representation,
    sha256_bytes,
    validate_passages,
)
from .passage_scoring import (
    CrossEncoderPairLengthFunction,
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
)
from .passage_score_cache import (
    CachedPassageScorer,
    PassageScoreCache,
)
from .reranking import (
    PassageEvidence,
    RerankedDocument,
    RerankingResult,
    rerank_retrieval,
)
from .paths import (
    PROJECT_CACHE_DIR_ENV,
    default_project_cache_dir,
    ensure_private_cache_dir,
)
from .flash_passage_scorer import (
    FlashEndpointBatchRunner,
    RunpodFlashConfig,
    RunpodFlashPassageScorer,
)
from .flash_worker import FlashPassageWorker, build_flash_worker
from .retriever import (
    DEFAULT_MAX_CONTENT_FALLBACKS,
    DocumentCorpus,
    RagtimeRetriever,
)

__all__ = [
    "Boundary",
    "Passage",
    "PassageChunker",
    "PassageChunkingError",
    "Representation",
    "sha256_bytes",
    "validate_passages",
    "CrossEncoderPairLengthFunction",
    "LocalCrossEncoderConfig",
    "LocalCrossEncoderScorer",
    "CachedPassageScorer",
    "PassageScoreCache",
    "PassageEvidence",
    "RerankedDocument",
    "RerankingResult",
    "rerank_retrieval",
    "FlashEndpointBatchRunner",
    "RunpodFlashConfig",
    "RunpodFlashPassageScorer",
    "FlashPassageWorker",
    "build_flash_worker",
]
