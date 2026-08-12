"""Ephemeral Flash definition for the deadline cold/warm smoke."""

import os

from runpod_flash import Endpoint

from trec_ragtime.flash_passage_scorer import RunpodFlashConfig
from trec_ragtime.flash_worker import build_flash_worker


_forwarded = {
    name: value
    for name in (
        "HF_ENDPOINT",
        "HF_TOKEN",
        "RAGTIME_ARTIFACT_CACHE_DIR",
        "RAGTIME_ARTIFACT_PREFIX",
        "RAGTIME_HF_BUCKET_ID",
    )
    if (value := os.environ.get(name))
}


def _endpoint_factory(**kwargs: object) -> Endpoint:
    return Endpoint(**kwargs, env=_forwarded)


_config = RunpodFlashConfig(
    endpoint_id="definition-only",
    batch_size=32,
    device="cuda",
    precision="float16",
    # Integer shorthand means (0, 1): never keep an idle worker billing.
    workers=1,
    max_concurrency=1,
    idle_timeout=300,
    worker_name="trec-ragtime-score-cache-smoke",
    gpu="AMPERE_24",
)

FlashPassageWorker = build_flash_worker(
    _config,
    endpoint_factory=_endpoint_factory,
)
