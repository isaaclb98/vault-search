"""
search/qdrant_client.py

Thin wrapper around the official qdrant-client. Read-only on the
search side — collection creation lives in the indexer.

Why a wrapper at all: tests want to swap in a mock that doesn't
require an actual qdrant-client connection. The wrapper exposes
a small surface (search, retrieve, ensure_collection,
healthz) that tests can mock cleanly.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Qdrant 1.18 requires point IDs to be UUIDs or unsigned integers.
# IMPLEMENTATION_DESIGN.md §"Qdrant schema" specifies the public
# point ID as the vault-relative path string; we satisfy Qdrant's
# constraint by deriving a deterministic UUID5 from the path. Same
# path → same UUID → idempotent upserts. The original path string
# is preserved in the payload's `path` field and in the wire-level
# FragmentHit.id (we convert back from UUID to the canonical path
# via the payload when reading). See "Deviations from
# IMPLEMENTATION_DESIGN.md" in the project README.
_POINT_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # URL


def point_id_for(vault_ref: str) -> str:
    """Deterministic UUID5 string for a vault-relative path."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, vault_ref))


@dataclass
class FragmentHit:
    id: str
    path: str
    content: str
    type: str
    filename: str
    score: float


class QdrantFragments:
    """
    Read-mostly wrapper around QdrantClient for the search side.

    For tests, swap the client out via the `client` attribute or
    pass a mock. The in-memory Qdrant (location=':memory:') works
    as a drop-in for local verification.
    """

    def __init__(self, client: Any, collection: str, timeout_ms: int = 5000):
        self.client = client
        self.collection = collection
        self.timeout_ms = timeout_ms

    def ensure_collection(self, vector_dim: int) -> None:
        """
        Create the collection if it doesn't exist. v0 schema: single
        768-dim cosine-named-vector, no payload indexes (small corpus).

        Called by the chat route on first tool dispatch if the
        collection is missing. Idempotent — does nothing if the
        collection already exists.
        """
        from qdrant_client.http import models as qmodels

        try:
            existing = self.client.get_collections()
            names = {c.name for c in existing.collections}
            if self.collection in names:
                return
        except Exception as e:
            raise QdrantUnavailable(f"Qdrant unreachable: {e}") from e

        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except Exception as e:
            # Race: another process may have just created it. Treat
            # "already exists" as success.
            msg = str(e).lower()
            if "already exists" in msg or "exists" in msg:
                return
            raise QdrantUnavailable(f"Qdrant create_collection failed: {e}") from e

    def search(
        self,
        vector: list[float],
        k: int = 8,
        type_filter: str | None = None,
    ) -> list[FragmentHit]:
        """
        Top-K semantic search. Optional `type_filter` restricts the
        result set to a single fragment type (quotation | fact |
        thought) via a server-side payload filter.

        Returns a list of FragmentHit sorted by score descending.
        Empty list on no matches. Raises QdrantUnavailable on
        connection failure.

        Each FragmentHit's `id` is the vault-relative path
        (`payload.path`), NOT the internal UUID5. Callers should
        pass the path string to `retrieve()`.
        """
        from qdrant_client.http import models as qmodels

        must_conditions: list[Any] = []
        if type_filter:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="type",
                    match=qmodels.MatchValue(value=type_filter),
                )
            )
        query_filter = (
            qmodels.Filter(must=must_conditions) if must_conditions else None
        )

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:
            raise QdrantUnavailable(f"Qdrant search failed: {e}") from e

        hits: list[FragmentHit] = []
        for r in response.points:
            payload = r.payload or {}
            # Prefer the canonical path string from the payload so
            # downstream callers see vault-relative paths, not UUIDs.
            canonical = str(payload.get("path") or r.id)
            hits.append(
                FragmentHit(
                    id=canonical,
                    path=canonical,
                    content=str(payload.get("content", "")),
                    type=str(payload.get("type", "")),
                    filename=str(payload.get("filename", "")),
                    score=float(r.score) if r.score is not None else 0.0,
                )
            )
        return hits

    def retrieve(self, ref: str) -> FragmentHit | None:
        """
        Look up a single point by vault-relative path. Returns None if
        missing.

        `ref` is the vault-relative path string. We translate it to
        the internal UUID5 to match Qdrant's storage key, but the
        returned FragmentHit.id is the canonical path.
        """
        internal_id = point_id_for(ref)
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=[internal_id],
                with_payload=True,
                with_vectors=False,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:
            raise QdrantUnavailable(f"Qdrant retrieve failed: {e}") from e
        if not points:
            return None
        p = points[0]
        payload = p.payload or {}
        canonical = str(payload.get("path") or ref)
        return FragmentHit(
            id=canonical,
            path=canonical,
            content=str(payload.get("content", "")),
            type=str(payload.get("type", "")),
            filename=str(payload.get("filename", "")),
            score=0.0,
        )

    def upsert(self, ref: str, vector: list[float], payload: dict[str, Any]) -> None:
        """
        Insert/replace a single point by vault-relative path. Used by
        the indexer. v0 doesn't batch — keep the surface small and
        obvious.

        `ref` is the public vault-relative path; the actual Qdrant
        point id is `point_id_for(ref)`. The payload's `path` field
        is set to `ref` so reads can recover the canonical path.
        """
        from qdrant_client.http import models as qmodels

        internal_id = point_id_for(ref)
        # Always set `path` to the canonical ref so the round-trip
        # is symmetric even if the caller forgot to set it.
        full_payload = dict(payload)
        full_payload.setdefault("path", ref)
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    qmodels.PointStruct(
                        id=internal_id,
                        vector=vector,
                        payload=full_payload,
                    )
                ],
                wait=True,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:
            raise QdrantUnavailable(f"Qdrant upsert failed: {e}") from e

    def healthz(self) -> bool:
        """Returns True if Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


class QdrantUnavailable(Exception):
    """Raised when Qdrant can't be reached or returns a fatal error."""