import logging

from pyobs.events import NewImageEvent
from pyobs.images.processor import ImageProcessor
from pyobs.images import Image
from pyobs.utils.fits import FilenameFormatter

log = logging.getLogger(__name__)


class Broadcast(ImageProcessor):
    """Broadcast image."""
    __module__ = 'pyobs.images.processors.misc'

    def __init__(self, filename: str = '/cache/{ORIGNAME}', *args, **kwargs):
        """Init an image processor that broadcasts an image

        Args:
            filename: Filename to broadcast image.
        """
        ImageProcessor.__init__(self, *args, **kwargs)

        # store
        self._formatter = FilenameFormatter(filename)

    def open(self):
        """Initialize processor."""
        ImageProcessor.open(self)

        # register event
        if self.comm is not None:
            self.comm.register_event(NewImageEvent)

    def __call__(self, image: Image) -> Image:
        """Broadcast image.

        Args:
            image: Image to broadcast.

        Returns:
            Original image.
        """

        # format filename
        filename = image.format_filename(self._formatter)

        # upload
        self.vfs.write_image(filename, image)

        # broadcast
        self.comm.send_event(NewImageEvent(filename))

        # finished
        return image


__all__ = ['Broadcast']
