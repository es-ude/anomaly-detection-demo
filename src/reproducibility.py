import subprocess
from typing import Optional


def get_commit_hash() -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None
