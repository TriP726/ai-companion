"""Shared Private-mode enforcement for services that persist data.

Three services now need identical privacy semantics (LLM, Memory, Graph).
Implementing it three times invites three subtly different bugs, so the rule
lives here once.

THE RULE
--------
A record created while the Vault is in PRIVATE mode is stamped `private=True`
at creation and is never written to disk. It remains fully usable in memory
for the life of the session and disappears on restart.

WHAT THIS DELIBERATELY DOES *NOT* DO
------------------------------------
Entering Private mode does not purge pre-existing memories or graph nodes from
disk. Those are long-term, owner-curated records that the user explicitly
approved in Normal mode; deleting them because the user toggled a mode would
be catastrophic, irreversible data loss.

This differs from conversations on purpose. A conversation is a single live
session: if you continue it in Private mode, the earlier half must be scrubbed,
because the badge would otherwise claim protection for a chat that is half on
disk. A memory is not a session — it is a stored fact with its own lifetime.

FAILURE POSTURE
---------------
`is_private_mode()` fails CLOSED. If the Vault service errors when queried we
assume Private. Wrongly skipping a write is recoverable; wrongly writing
private data to disk is not.
"""
from __future__ import annotations

from typing import Any, Protocol


class _HasServiceManager(Protocol):
    _service_manager: Any


class PrivacyMixin:
    """Gives a service Vault-aware privacy checks.

    Requires the host to be a BaseService (for `_service_manager`, set by
    ServiceManager.register()).
    """

    def is_private_mode(self) -> bool:
        """True when the Vault is in PRIVATE mode.

        Returns False when no ServiceManager is wired at all — that means the
        vault feature is absent (unit-test context), not failing. Returns True
        on any error while querying, which is the fail-closed path.
        """
        manager = getattr(self, "_service_manager", None)
        if manager is None:
            return False
        try:
            vault = manager.get("Vault")
            if vault is None:
                return False
            return bool(vault.is_private)
        except Exception:  # noqa: BLE001 - fail closed, never raise
            return True

    def privacy_stamp(self) -> bool:
        """Value for a new record's `private` field."""
        return self.is_private_mode()


def save_public_only(store: Any, records: dict[str, dict]) -> None:
    """Persist only records whose `private` flag is falsy.

    `store` keeps every record in memory so the session sees them all; only
    the disk write is filtered. Private records are actively removed from the
    on-disk copy, so a record that somehow reached disk before being marked
    private does not linger.
    """
    public = {k: v for k, v in records.items() if not v.get("private")}
    store.clear()
    for key, value in public.items():
        store.set(key, value)
    store.save()
    # Restore the full in-memory view so the running session is unaffected.
    for key, value in records.items():
        if value.get("private"):
            store.set(key, value)


def count_private(records: dict[str, dict]) -> int:
    return sum(1 for v in records.values() if v.get("private"))
