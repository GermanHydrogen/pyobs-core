from typing import Tuple

from .interface import Interface


class ICoordsHGS(Interface):
    """The module can move to Mu/Psi coordinates, usually combined with :class:`~pyobs.interfaces.ITelescope`."""
    __module__ = 'pyobs.interfaces'

    def move_hgs_lon_lat(self, lon: float, lat: float, *args, **kwargs):
        """Moves on given coordinates.

        Args:
            lon: Longitude in deg to track.
            lat: Latitude in deg to track.

        Raises:
            ValueError: If device could not move.
        """
        raise NotImplementedError

    def get_hgs_lon_lat(self, *args, **kwargs) -> Tuple[float, float]:
        """Returns current longitude and latitude position.

        Returns:
            Tuple of current lon, lat in degrees.
        """
        raise NotImplementedError


__all__ = ['ICoordsHGS']
