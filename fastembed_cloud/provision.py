"""
fastembed-cloud — Auto-Provisioning

Zero-friction cloud provisioning for embedding users.
On first use, if no API key is found, automatically provisions
a free AINative account (72-hour TTL) that can be claimed later.

Credential resolution order:
1. Explicit api_key parameter
2. AINATIVE_API_KEY environment variable
3. ZERODB_API_KEY environment variable (shared with zerodb ecosystem)
4. ~/.zerodb/credentials.json (shared credential store)
5. Auto-provision via /api/v1/public/instant-db

Refs #3943
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ZERODB_DIR = Path.home() / ".zerodb"
CREDS_PATH = ZERODB_DIR / "credentials.json"
CONFIG_PATH = ZERODB_DIR / "config.json"
CLOUD_API_URL = "https://api.ainative.studio"
PROVISION_ENDPOINT = "/api/v1/public/instant-db"


def resolve_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Resolve an API key from all available sources.

    Args:
        explicit_key: Key passed directly by the user.

    Returns:
        A valid API key string.

    Raises:
        RuntimeError: If auto-provisioning fails and no key is available.
    """
    # 1. Explicit parameter
    if explicit_key:
        return explicit_key

    # 2. Environment variables
    env_key = (
        os.environ.get("AINATIVE_API_KEY")
        or os.environ.get("ZERODB_API_KEY")
    )
    if env_key:
        return env_key

    # 3. Credentials file (shared with zerodb ecosystem)
    creds = _load_credentials()
    if creds:
        return creds

    # 4. Auto-provision
    return _auto_provision()


def _load_credentials() -> Optional[str]:
    """Load API key from ~/.zerodb/credentials.json."""
    if CREDS_PATH.exists():
        try:
            data = json.loads(CREDS_PATH.read_text())
            key = data.get("api_key")
            if key:
                return key
        except (json.JSONDecodeError, KeyError):
            pass

    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            key = data.get("api_key") or data.get("cloud_api_key")
            if key:
                return key
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def _auto_provision() -> str:
    """
    Auto-provision a free AINative account for embedding access.

    Returns:
        API key string.

    Raises:
        RuntimeError: If provisioning fails.
    """
    print(
        "\n  No API key found — provisioning a free AINative account for embeddings...",
        file=sys.stderr,
    )

    import requests

    try:
        resp = requests.post(
            f"{CLOUD_API_URL}{PROVISION_ENDPOINT}",
            json={"agree_terms": True, "source": "fastembed-cloud"},
            timeout=15,
        )

        if resp.status_code == 429:
            raise RuntimeError(
                "Rate limited — too many provisions from this IP. "
                "Sign up at https://ainative.studio/signup and set AINATIVE_API_KEY."
            )

        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Provisioning failed (HTTP {resp.status_code}). "
                "Sign up at https://ainative.studio/signup and set AINATIVE_API_KEY."
            )

        data = resp.json()
        api_key = data.get("api_key", "")
        if not api_key:
            raise RuntimeError("Provisioning returned empty API key.")

        # Save credentials for reuse
        _save_credentials(data)
        _print_success(data)
        return api_key

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Network error during provisioning: {exc}. "
            "Set AINATIVE_API_KEY manually or sign up at https://ainative.studio/signup"
        ) from exc


def _save_credentials(data: dict) -> None:
    """Save provisioned credentials to ~/.zerodb/ for ecosystem sharing."""
    ZERODB_DIR.mkdir(parents=True, exist_ok=True)

    creds = {
        "api_key": data.get("api_key", ""),
        "project_id": data.get("project_id", ""),
        "base_url": data.get("base_url", CLOUD_API_URL),
        "expires_at": data.get("expires_at", ""),
        "claim_url": data.get("claim_url", ""),
        "provisioned_at": datetime.utcnow().isoformat(),
        "source": "fastembed-cloud",
    }

    CREDS_PATH.write_text(json.dumps(creds, indent=2) + "\n")


def _print_success(data: dict) -> None:
    """Print success message with claim URL."""
    expires = data.get("expires_at", "72 hours")
    claim_url = data.get("claim_url", "https://ainative.studio/signup")
    api_key = data.get("api_key", "")

    print(
        f"\n  Auto-provisioned! Free embeddings API ready.\n"
        f"\n"
        f"  API Key:   {api_key[:12]}...\n"
        f"  Expires:   {expires}\n"
        f"  Saved to:  ~/.zerodb/credentials.json\n"
        f"\n"
        f"  To keep access permanently, claim your account:\n"
        f"  {claim_url}\n",
        file=sys.stderr,
    )
