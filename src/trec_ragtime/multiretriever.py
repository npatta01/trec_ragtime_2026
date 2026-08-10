"""Multi-query retrieval, deterministic fusion, and one-pass hydration."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .fusion import (
    FusedHit,
    FusedRanking,
    QueryRanking,
    RetrievalQuery,
    weighted_reciprocal_rank_fusion,
)
from .models import HuggingFaceDocument, HydrationFailure
from .retriever import RagtimeRetriever


class MultiQueryRetrievalError(RuntimeError):
    """Raised when a fused result cannot safely feed report synthesis."""


@dataclass(frozen=True, slots=True)
class FusedDocument:
    """One fused hit joined to its canonical corpus record."""

    hit: FusedHit
    document: HuggingFaceDocument

    def __post_init__(self) -> None:
        if self.hit.document_id != self.document.document_id:
            raise ValueError("fused hit and hydrated document IDs do not match")

    @property
    def rank(self) -> int:
        return self.hit.rank

    @property
    def document_id(self) -> str:
        return self.hit.document_id

    @property
    def score(self) -> float:
        return self.hit.score

    @property
    def language(self) -> str:
        return self.document.language

    @property
    def text(self) -> str:
        """Return canonical source-language text."""

        return self.document.text

    @property
    def report_text(self) -> str | None:
        """Return the verified English synthesis representation."""

        return self.document.report_text


@dataclass(frozen=True, slots=True)
class MultiQueryRetrievalResult:
    """Search rankings, their fused order, and the hydrated generation pool."""

    rankings: tuple[QueryRanking, ...]
    fused_ranking: FusedRanking
    documents: tuple[FusedDocument, ...]
    hydrated_hit_count: int
    unresolved_ids: tuple[str, ...] = ()
    hydration_failures: tuple[HydrationFailure, ...] = ()
    missing_translation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            rankings = tuple(self.rankings)
        except TypeError as error:
            raise ValueError("rankings must be an iterable of QueryRanking") from error
        if any(not isinstance(ranking, QueryRanking) for ranking in rankings):
            raise ValueError("rankings must contain only QueryRanking values")
        if not isinstance(self.fused_ranking, FusedRanking):
            raise ValueError("fused_ranking must be a FusedRanking")
        if tuple(ranking.query for ranking in rankings) != self.fused_ranking.queries:
            raise ValueError("rankings do not match the fused ranking queries")

        if (
            isinstance(self.hydrated_hit_count, bool)
            or not isinstance(self.hydrated_hit_count, int)
            or self.hydrated_hit_count < 0
            or self.hydrated_hit_count > len(self.fused_ranking.hits)
        ):
            raise ValueError("hydrated_hit_count is outside the fused ranking")
        try:
            documents = tuple(self.documents)
        except TypeError as error:
            raise ValueError(
                "documents must be an iterable of FusedDocument"
            ) from error
        if any(not isinstance(document, FusedDocument) for document in documents):
            raise ValueError("documents must contain only FusedDocument values")
        requested_hits = self.fused_ranking.hits[: self.hydrated_hit_count]
        expected_ids = {hit.document_id for hit in requested_hits}
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("documents contain duplicate IDs")
        if any(document_id not in expected_ids for document_id in document_ids):
            raise ValueError("document was not requested from the fused ranking")
        document_id_set = set(document_ids)
        unresolved = tuple(
            hit.document_id
            for hit in requested_hits
            if hit.document_id not in document_id_set
        )
        if isinstance(self.unresolved_ids, (str, bytes)):
            raise ValueError("unresolved_ids must be an iterable of strings")
        try:
            supplied_unresolved = tuple(self.unresolved_ids)
        except TypeError as error:
            raise ValueError("unresolved_ids must be an iterable of strings") from error
        if unresolved != supplied_unresolved:
            raise ValueError("unresolved_ids do not match the hydrated document set")
        expected_documents = tuple(
            hit for hit in requested_hits if hit.document_id in document_id_set
        )
        if tuple(document.hit for document in documents) != expected_documents:
            raise ValueError("documents must follow the hydrated fused-prefix order")

        try:
            failures = tuple(self.hydration_failures)
        except TypeError as error:
            raise ValueError(
                "hydration_failures must be an iterable of HydrationFailure"
            ) from error
        if any(not isinstance(failure, HydrationFailure) for failure in failures):
            raise ValueError(
                "hydration_failures must contain only HydrationFailure values"
            )
        if isinstance(self.missing_translation_ids, (str, bytes)):
            raise ValueError(
                "missing_translation_ids must be an iterable of strings"
            )
        try:
            missing_translation_ids = tuple(self.missing_translation_ids)
        except TypeError as error:
            raise ValueError(
                "missing_translation_ids must be an iterable of strings"
            ) from error
        if len(missing_translation_ids) != len(set(missing_translation_ids)):
            raise ValueError("missing_translation_ids must not contain duplicates")
        missing_translation_set = set(missing_translation_ids)
        if any(
            not isinstance(document_id, str) or document_id not in document_id_set
            for document_id in missing_translation_ids
        ):
            raise ValueError(
                "missing_translation_ids must refer to hydrated documents"
            )
        expected_missing_order = tuple(
            hit.document_id
            for hit in requested_hits
            if hit.document_id in missing_translation_set
        )
        if missing_translation_ids != expected_missing_order:
            raise ValueError(
                "missing_translation_ids must follow the fused-prefix order"
            )

        object.__setattr__(self, "rankings", rankings)
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "unresolved_ids", supplied_unresolved)
        object.__setattr__(self, "hydration_failures", failures)
        object.__setattr__(
            self,
            "missing_translation_ids",
            missing_translation_ids,
        )

    @property
    def fully_hydrated(self) -> bool:
        return (
            not self.unresolved_ids
            and len(self.documents) == self.hydrated_hit_count
        )

    @property
    def fully_report_ready(self) -> bool:
        return (
            self.hydrated_hit_count > 0
            and self.fully_hydrated
            and not self.hydration_failures
            and not self.missing_translation_ids
            and all(document.report_text is not None for document in self.documents)
        )

    def require_report_ready(self) -> None:
        """Fail before generation when canonical English evidence is incomplete."""

        if self.fully_report_ready:
            return
        details: list[str] = []
        if self.unresolved_ids:
            details.append(f"{len(self.unresolved_ids)} unresolved document IDs")
        if self.missing_translation_ids:
            details.append(
                f"{len(self.missing_translation_ids)} missing English translations"
            )
        if self.hydration_failures:
            details.append(f"{len(self.hydration_failures)} hydration failures")
        if not details:
            details.append("one or more documents lack report-ready text")
        raise MultiQueryRetrievalError(
            "fused retrieval is not ready for English report synthesis: "
            + "; ".join(details)
        )


class MultiQueryRetriever:
    """Execute a query plan, fuse its ranks, then hydrate each ID only once."""

    def __init__(self, retriever: RagtimeRetriever) -> None:
        if not isinstance(retriever, RagtimeRetriever):
            raise TypeError("retriever must be a RagtimeRetriever")
        self.retriever = retriever

    def retrieve(
        self,
        queries: Iterable[RetrievalQuery],
        *,
        rrf_k: float = 60.0,
        max_documents: int = 1000,
        hydrate_top_k: int | None = None,
        refresh_search: bool = False,
        refresh_documents: bool = False,
    ) -> MultiQueryRetrievalResult:
        """Search every planned query and hydrate the fused prefix.

        ``max_documents`` bounds the globally fused MLIR ranking.
        ``hydrate_top_k`` can select a smaller prefix for report generation
        without weakening or changing that submitted ranking.
        """

        planned_queries = tuple(queries)
        if not planned_queries:
            raise ValueError("queries must not be empty")
        if any(not isinstance(query, RetrievalQuery) for query in planned_queries):
            raise TypeError("queries must contain RetrievalQuery values")
        query_ids = tuple(query.query_id for query in planned_queries)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query IDs must be unique")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, (int, float)):
            raise ValueError("rrf_k must be a positive finite number")
        try:
            numeric_rrf_k = float(rrf_k)
        except OverflowError as error:
            raise ValueError("rrf_k must be a positive finite number") from error
        if not math.isfinite(numeric_rrf_k) or numeric_rrf_k <= 0:
            raise ValueError("rrf_k must be a positive finite number")
        if (
            isinstance(max_documents, bool)
            or not isinstance(max_documents, int)
            or not 1 <= max_documents <= 1000
        ):
            raise ValueError("max_documents must be an integer from 1 to 1000")
        if hydrate_top_k is not None and (
            isinstance(hydrate_top_k, bool)
            or not isinstance(hydrate_top_k, int)
            or hydrate_top_k < 1
            or hydrate_top_k > max_documents
        ):
            raise ValueError(
                "hydrate_top_k must be a positive integer no larger than max_documents"
            )
        for name, value in (
            ("refresh_search", refresh_search),
            ("refresh_documents", refresh_documents),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

        rankings = tuple(
            QueryRanking(
                query=query,
                result=self.retriever.client.search(
                    query.text,
                    pipeline=query.pipeline,
                    refresh=refresh_search,
                ),
            )
            for query in planned_queries
        )
        fused = weighted_reciprocal_rank_fusion(
            rankings,
            k=numeric_rrf_k,
            max_docs=max_documents,
        )
        hydrated_hit_count = min(
            len(fused.hits),
            max_documents if hydrate_top_k is None else hydrate_top_k,
        )
        requested_hits = fused.hits[:hydrated_hit_count]
        hydration = self.retriever.enrich_documents(
            (hit.document_id for hit in requested_hits),
            refresh_documents=refresh_documents,
        )
        requested_ids = tuple(hit.document_id for hit in requested_hits)
        requested_id_set = set(requested_ids)
        hydrated_ids = tuple(
            document.document_id for document in hydration.documents
        )
        if len(hydrated_ids) != len(set(hydrated_ids)):
            raise MultiQueryRetrievalError(
                "corpus hydration returned duplicate document IDs"
            )
        if any(document_id not in requested_id_set for document_id in hydrated_ids):
            raise MultiQueryRetrievalError(
                "corpus hydration returned an unrequested document ID"
            )
        hydrated_id_set = set(hydrated_ids)
        expected_unresolved = tuple(
            document_id
            for document_id in requested_ids
            if document_id not in hydrated_id_set
        )
        if tuple(hydration.unresolved_ids) != expected_unresolved:
            raise MultiQueryRetrievalError(
                "corpus hydration returned inconsistent unresolved IDs"
            )
        if any(
            document_id not in hydrated_id_set
            for document_id in hydration.missing_translation_ids
        ):
            raise MultiQueryRetrievalError(
                "corpus hydration reported a translation miss for an "
                "unhydrated document"
            )
        by_id = hydration.by_id()
        documents = tuple(
            FusedDocument(hit=hit, document=by_id[hit.document_id])
            for hit in requested_hits
            if hit.document_id in by_id
        )
        unresolved = tuple(
            hit.document_id
            for hit in requested_hits
            if hit.document_id not in by_id
        )
        missing_translation_id_set = set(hydration.missing_translation_ids)
        missing_translation_ids = tuple(
            hit.document_id
            for hit in requested_hits
            if hit.document_id in missing_translation_id_set
        )
        return MultiQueryRetrievalResult(
            rankings=rankings,
            fused_ranking=fused,
            documents=documents,
            hydrated_hit_count=hydrated_hit_count,
            unresolved_ids=unresolved,
            hydration_failures=hydration.failures,
            missing_translation_ids=missing_translation_ids,
        )


__all__ = [
    "FusedDocument",
    "MultiQueryRetrievalError",
    "MultiQueryRetrievalResult",
    "MultiQueryRetriever",
]
