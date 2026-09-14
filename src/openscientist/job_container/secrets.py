"""Per-job secret derivation for the untrusted agent container."""

from __future__ import annotations

import hashlib
import hmac

_JOB_SECRET_LABEL = "job_secret:"
_LLM_PROXY_LABEL = "llm_proxy:"
_EXEC_TOKEN_LABEL = "exec_token:"
_PLACEHOLDER_SEP = "."


def derive_job_secret(master_key: str, job_id: str) -> str:
    """Return HMAC-SHA256(master_key, "job_secret:" + job_id) as a 64-char hex digest.

    Keeps the master key out of the job container while satisfying the required
    Settings.secret_key field inside it.
    """
    message = f"{_JOB_SECRET_LABEL}{job_id}".encode()
    return hmac.new(master_key.encode(), message, hashlib.sha256).hexdigest()


def derive_llm_proxy_token(master_key: str, job_id: str) -> str:
    """Per-job token presented to the LLM proxy, keyed by a distinct label."""
    message = f"{_LLM_PROXY_LABEL}{job_id}".encode()
    return hmac.new(master_key.encode(), message, hashlib.sha256).hexdigest()


def make_job_placeholder(master_key: str, job_id: str) -> str:
    """Placeholder LLM credential: "<job_id>.<token>", verifiable by the proxy."""
    return f"{job_id}{_PLACEHOLDER_SEP}{derive_llm_proxy_token(master_key, job_id)}"


def verify_job_placeholder(master_key: str, placeholder: str) -> bool:
    """True when placeholder is a valid "<job_id>.<token>" for the master key."""
    job_id, sep, token = placeholder.rpartition(_PLACEHOLDER_SEP)
    if sep != _PLACEHOLDER_SEP or not job_id or not token:
        return False
    return hmac.compare_digest(derive_llm_proxy_token(master_key, job_id), token)


def derive_exec_token(master_key: str, job_id: str) -> str:
    """Per-job token the execution broker verifies, keyed by a distinct label."""
    message = f"{_EXEC_TOKEN_LABEL}{job_id}".encode()
    return hmac.new(master_key.encode(), message, hashlib.sha256).hexdigest()


def make_exec_placeholder(master_key: str, job_id: str) -> str:
    """Execution credential "<job_id>.<token>" the broker recomputes and verifies."""
    return f"{job_id}{_PLACEHOLDER_SEP}{derive_exec_token(master_key, job_id)}"
