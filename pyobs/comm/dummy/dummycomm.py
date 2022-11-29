import inspect
import logging
from typing import Any, List, Type

from pyobs.comm import Comm
from pyobs.events import Event
from pyobs.interfaces import Interface
from pyobs.utils.parallel import Future

log = logging.getLogger(__name__)


class DummyComm(Comm):
    """A dummy implementation of the Comm interface."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Creates a new dummy comm."""
        Comm.__init__(self, *args, **kwargs)

    @property
    def clients(self) -> List[str]:
        """Always return zero clients."""
        return []

    def get_interfaces(self, client: str) -> List[Type[Interface]]:
        """No interfaces implemented."""
        return []

    def _supports_interface(self, client: str, interface: Type[Interface]) -> bool:
        """Interfaces are never supported."""
        return False

    def execute(self, client: str, method: str, signature: inspect.Signature, *args: Any) -> Future:
        """Always fake a successful execution of a method."""
        return Future(empty=True)

    @property
    def name(self) -> str:
        """Name of this client, which is unknown."""
        return "module"

    async def send_event(self, event: Event) -> None:
        """Send an event to other clients.

        Args:
            event (Event): Event to send
        """
        self._send_event_to_module(event, "module")


__all__ = ["DummyComm"]
