"""Versioned H100 throughput envelopes used only as anomaly signals."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

from gpuforge.performance import PerformanceResult, Precision, WorkUnit

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_FAMILY = re.compile(r"[a-z][a-z0-9_]{0,31}")
_H100_CLASSES = frozenset({"h100_sxm", "h100_pcie", "h100_nvl"})


class BaselineDecision(str, Enum):
    """Conservative anomaly outcome; neither result proves GPU identity."""

    PLAUSIBLE = "plausible"
    SLOW_REVIEW = "slow_review"
    IMPOSSIBLE = "impossible"
    UNPROFILED = "unprofiled"


@dataclass(frozen=True, slots=True)
class BaselineProfile:
    """Reviewed throughput envelope for one workload and GPU configuration."""

    version: int
    workload_family: str
    model_shape_digest: str
    gpu_class: str
    gpu_count: int
    precision: Precision
    unit: WorkUnit
    lower_milli_units_per_second: int
    upper_milli_units_per_second: int
    reviewer_signature: str

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("invalid_baseline_profile")
        if not isinstance(self.workload_family, str) or not _FAMILY.fullmatch(self.workload_family):
            raise ValueError("invalid_baseline_profile")
        if not isinstance(self.model_shape_digest, str) or not _DIGEST.fullmatch(
            self.model_shape_digest
        ):
            raise ValueError("invalid_baseline_profile")
        if self.gpu_class not in _H100_CLASSES or not isinstance(self.precision, Precision):
            raise ValueError("invalid_baseline_profile")
        if not isinstance(self.unit, WorkUnit) or isinstance(self.gpu_count, bool):
            raise ValueError("invalid_baseline_profile")
        if not 1 <= self.gpu_count <= 8:
            raise ValueError("invalid_baseline_profile")
        for value in (self.lower_milli_units_per_second, self.upper_milli_units_per_second):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("invalid_baseline_profile")
        if self.lower_milli_units_per_second > self.upper_milli_units_per_second:
            raise ValueError("invalid_baseline_profile")
        if not isinstance(self.reviewer_signature, str) or not re.fullmatch(
            r"[0-9a-f]{128}", self.reviewer_signature
        ):
            raise ValueError("invalid_baseline_profile")

    def signing_bytes(self) -> bytes:
        """Return canonical profile data to be approved by a trusted reviewer."""
        value = {
            "gpu_class": self.gpu_class,
            "gpu_count": self.gpu_count,
            "lower": self.lower_milli_units_per_second,
            "model_shape_digest": self.model_shape_digest,
            "precision": self.precision.value,
            "unit": self.unit.value,
            "upper": self.upper_milli_units_per_second,
            "version": self.version,
            "workload_family": self.workload_family,
        }
        return b"gpuforge-baseline-v1\x00" + json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("ascii")

    def digest(self) -> str:
        """Identify reviewed content and its signature without recording a host."""
        return (
            "sha256:"
            + hashlib.sha256(
                self.signing_bytes() + bytes.fromhex(self.reviewer_signature)
            ).hexdigest()
        )


def assess_baseline(
    profile: BaselineProfile | None,
    result: PerformanceResult,
    *,
    workload_family: str,
    gpu_class: str,
    required_version: int,
    review_verified: bool,
    mig_enabled: bool = False,
) -> BaselineDecision:
    """Classify throughput; slow correct miners are flagged, not rejected."""
    if mig_enabled or gpu_class not in _H100_CLASSES:
        return BaselineDecision.UNPROFILED
    if profile is None or not review_verified or profile.version != required_version:
        return BaselineDecision.UNPROFILED
    if (
        profile.workload_family != workload_family
        or profile.gpu_class != gpu_class
        or profile.gpu_count != result.gpu_count
        or profile.precision is not result.precision
        or profile.unit is not result.unit
        or profile.model_shape_digest != result.model_shape_digest
    ):
        return BaselineDecision.UNPROFILED
    if result.rate_milli_units_per_second > profile.upper_milli_units_per_second:
        return BaselineDecision.IMPOSSIBLE
    if result.rate_milli_units_per_second < profile.lower_milli_units_per_second:
        return BaselineDecision.SLOW_REVIEW
    return BaselineDecision.PLAUSIBLE
