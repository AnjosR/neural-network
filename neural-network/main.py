"""Ponto de entrada do projeto."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    ...


if __name__ == "__main__":
    main()
