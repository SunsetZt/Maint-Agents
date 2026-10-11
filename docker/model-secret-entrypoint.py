"""Wait for Vault-managed injection; exec the service without exposing secrets."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import time


def credential_ready(path: Path) -> bool:
    try:
        metadata = path.lstat()
        return (
            stat.S_ISREG(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o400
            and metadata.st_uid == 65532
            and 0 < metadata.st_size <= 4096
            and os.access(path, os.R_OK)
            and (path.parent / "ready").is_file()
        )
    except OSError:
        return False


def main() -> None:
    path = Path(os.environ["IOAP_MODEL_GATEWAY_API_KEY_FILE"])
    if not path.is_absolute() or not sys.argv[1:]:
        raise SystemExit("MODEL_SECRET_CONFIGURATION_INVALID")
    # No HTTP listener or GPU weights are started before injection is available.
    # Remain restartable when Vault is slower than Docker during cold startup.
    if not credential_ready(path):
        print("Waiting for Vault model secret injection.", flush=True)
    while not credential_ready(path):
        time.sleep(1)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
