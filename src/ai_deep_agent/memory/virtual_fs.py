from __future__ import annotations
from typing import Optional


class VirtualFileSystem:
    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    def write(self, filename: str, content: str) -> None:
        self._files[filename] = content

    def read(self, filename: str) -> Optional[str]:
        return self._files.get(filename)

    def exists(self, filename: str) -> bool:
        return filename in self._files

    def list_files(self) -> list[str]:
        return list(self._files.keys())

    def read_all(self) -> dict[str, str]:
        return dict(self._files)

    def build_context(self, exclude: Optional[list[str]] = None) -> str:
        skip = set(exclude or [])
        parts = [
            f"=== {name} ===\n{content}"
            for name, content in self._files.items()
            if name not in skip
        ]
        return "\n\n".join(parts) if parts else "(workspace empty)"

    def word_count(self) -> int:
        return sum(len(v.split()) for v in self._files.values())

    def clear(self) -> None:
        self._files.clear()


workspace = VirtualFileSystem()
