from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Any, Type, List
from astroplan import ObservingBlock

from pyobs.utils.time import Time
from .task import Task
from pyobs.object import Object


class TaskArchive(Object, metaclass=ABCMeta):
    def __init__(self, **kwargs: Any):
        Object.__init__(self, **kwargs)

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def _create_task(self, klass: Type[Task], **kwargs: Any) -> Task:
        return self.get_object(klass, tasks=self, **kwargs)

    @abstractmethod
    async def last_changed(self) -> Optional[Time]:
        """Returns time when last time any blocks changed."""
        ...

    @abstractmethod
    async def get_schedulable_blocks(self) -> List[ObservingBlock]:
        """Returns list of schedulable blocks.

        Returns:
            List of schedulable blocks
        """
        ...


__all__ = ["TaskArchive"]
