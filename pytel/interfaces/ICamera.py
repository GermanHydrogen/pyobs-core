import enum
from typing import Union

from .IStatus import IStatus
from .IAbortable import IAbortable


class ICamera(IStatus, IAbortable):
    """Basic interface for all cameras."""

    class CameraStatus(enum.Enum):
        """Enumerator for camera status."""
        IDLE = 'idle'
        EXPOSING = 'exposing'
        READOUT = 'readout'

    class ImageType(enum.Enum):
        """Enumerator specifying the image type."""
        BIAS = 'bias'
        DARK = 'dark'
        OBJECT = 'object'
        FLAT = 'flat'

    def get_status(self, *args, **kwargs) -> str:
        """Returns the current status of the camera, which is one of 'idle', 'exposing', or 'readout'.

        Returns:
            Current status of camera.
        """
        raise NotImplementedError

    def expose(self, exposure_time: int, image_type: str, count: int = 1, broadcast: bool = True,
               *args, **kwargs) -> Union[str, list]:
        """Starts exposure and returns reference to image.

        Args:
            exposure_time: Exposure time in seconds.
            image_type: Type of image.
            count: Number of images to take.
            broadcast: Broadcast existence of image.

        Returns:
            str/list: Reference to the image that was taken or list of references, if count>1.
        """
        raise NotImplementedError

    def abort(self, *args, **kwargs) -> bool:
        """Aborts the current exposure and sequence.

        Returns:
            bool: True if successful, otherwise False.
        """
        raise NotImplementedError

    def abort_sequence(self, *args, **kwargs) -> bool:
        """Aborts the current sequence after current exposure.

        Returns:
            bool: True if successful, otherwise False.
        """
        raise NotImplementedError

    def get_exposure_time_left(self, *args, **kwargs) -> float:
        """Returns the remaining exposure time on the current exposure in ms.

        Returns:
            Remaining exposure time in ms.
        """
        raise NotImplementedError

    def get_exposure_progress(self, *args, **kwargs) -> float:
        """Returns the progress of the current exposure in percent.

        Returns:
            Progress of the current exposure in percent.
        """
        raise NotImplementedError

    def status(self, *args, **kwargs) -> dict:
        """Returns current status of camera.

        Returns:
            A dictionary that should contain at least the following fields:

            ICamera
                status (str):               Current status of camera.
                ExposureTimeLeft (float):   Time in seconds left before finished current action (expose/readout).
                ExposuresLeft (int):        Number of remaining exposures.
                Progress (float):           Percentage of how much of current action (expose/readout) is finished.
                LastImage (str):            Reference to last image taken.
        """
        raise NotImplementedError


__all__ = ['ICamera']
