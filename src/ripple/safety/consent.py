"""Consent and speaker-profile authorization policy."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class ProfileUse(StrEnum):
    CONVERSION = "conversion"
    TTS = "tts"
    EXPORT = "export"


class ConsentDenied(PermissionError):
    pass


@dataclass(frozen=True)
class ConsentRecord:
    subject_id: str
    profile_id: str
    allowed_uses: frozenset[ProfileUse]
    granted_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    evidence_uri: str | None = None
    policy_version: str = "1"
    attributes: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.subject_id or not self.profile_id:
            raise ValueError("subject_id and profile_id are required")
        if self.granted_at.tzinfo is None:
            raise ValueError("granted_at must be timezone-aware")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.revoked_at is not None and self.revoked_at.tzinfo is None:
            raise ValueError("revoked_at must be timezone-aware")
        if self.expires_at is not None and self.expires_at <= self.granted_at:
            raise ValueError("expires_at must be after granted_at")


@dataclass(frozen=True)
class ProfilePolicy:
    """Fail-closed policy for production voice-profile use."""

    require_evidence: bool = True
    denied_subjects: frozenset[str] = frozenset()
    allowed_policy_versions: frozenset[str] = frozenset({"1"})

    def authorize(
        self,
        consent: ConsentRecord,
        use: ProfileUse,
        *,
        profile_id: str,
        now: datetime | None = None,
    ) -> None:
        consent.validate()
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        reasons: list[str] = []
        if consent.profile_id != profile_id:
            reasons.append("profile mismatch")
        if consent.subject_id in self.denied_subjects:
            reasons.append("subject is denied")
        if consent.policy_version not in self.allowed_policy_versions:
            reasons.append("unsupported policy version")
        if use not in consent.allowed_uses:
            reasons.append(f"{use.value} is not authorized")
        if consent.granted_at > current:
            reasons.append("consent is not yet active")
        if consent.expires_at is not None and current >= consent.expires_at:
            reasons.append("consent expired")
        if consent.revoked_at is not None and current >= consent.revoked_at:
            reasons.append("consent revoked")
        if self.require_evidence and not consent.evidence_uri:
            reasons.append("consent evidence is missing")
        if reasons:
            raise ConsentDenied("; ".join(reasons))


def parse_allowed_uses(values: Collection[str]) -> frozenset[ProfileUse]:
    try:
        return frozenset(ProfileUse(value) for value in values)
    except ValueError as exc:
        raise ValueError("allowed uses must be conversion, tts, or export") from exc
