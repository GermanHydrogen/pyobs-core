import fnmatch
from abc import ABCMeta, abstractmethod
from typing import Any, AnyStr, List


class VFSFile(metaclass=ABCMeta):
    """Base class for all VFS file classes."""

    __module__ = "pyobs.vfs"

    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def read(self, n: int = -1) -> AnyStr:
        ...

    @abstractmethod
    async def write(self, s: AnyStr) -> None:
        ...

    async def __aenter__(self) -> "VFSFile":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @staticmethod
    def listdir(path: str, **kwargs: Any) -> List[str]:
        """Returns content of given path.

        Args:
            path: Path to list.
            kwargs: Parameters for specific file implementation (same as __init__).

        Returns:
            List of files in path.
        """
        raise NotImplementedError()

    @classmethod
    def find(cls, path: str, pattern: str, **kwargs: Any) -> List[str]:
        """Find files by pattern matching.

        Args:
            path: Path to search in.
            pattern: Pattern to search for.

        Returns:
            List of found files.
        """

        # list files in dir
        files = cls.listdir(path, **kwargs)

        # filter by pattern
        return list(filter(lambda f: fnmatch.fnmatch(f, pattern), files))


__all__ = ["VFSFile"]
