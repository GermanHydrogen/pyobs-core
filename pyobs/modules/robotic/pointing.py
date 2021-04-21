import logging
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from pyobs.modules import Module
from pyobs.comm import InvocationException
from pyobs.interfaces import IAcquisition, IAutonomous, ITelescope, ICamera, ICameraExposureTime, ICameraBinning, \
    IImageType
from pyobs.utils.enums import ImageType
from pyobs.utils.time import Time

log = logging.getLogger(__name__)


class PointingSeries(Module, IAutonomous):
    """Module for running pointing series."""
    __module__ = 'pyobs.modules.robotic'

    def __init__(self, alt_range: Tuple[float, float] = (30., 85.), num_alt: int = 8,
                 az_range: Tuple[float, float] = (0., 360.), num_az: int = 24,
                 dec_range: Tuple[float, float] = (-80., 80.), min_moon_dist: float = 15., finish: int = 90,
                 randomize: bool = True, two_way: bool = False,
                 exp_time: float = 1., acquisition: str = None, telescope: str = 'telescope',
                 cameras: List[str] = None, acquire: bool = True, broadcast: bool = True, write_to: str = None,
                 *args, **kwargs):
        """Initialize a new auto focus system.

        Args:
            alt_range: Range in degrees to use in altitude.
            num_alt: Number of altitude points to create on grid.
            az_range: Range in degrees to use in azimuth.
            num_az: Number of azimuth points to create on grid.
            dec_range: Range in declination in degrees to use.
            min_moon_dist: Minimum moon distance in degrees.
            finish: When this number in percent of points have been finished, terminate mastermind.
            randomize: Randomize grid points.
            two_way: Only used, if randomize=False. Iterate all grid points again in reverse order.
            exp_time: Exposure time in secs.
            acquisition: IAcquisition unit to use.
            telescope: ITelescope unit to use.
            cameras: List of cameras to take images with.
            acquire: If True, the given acquisition module is used for fine-acquisition. Otherwise only images are
                taken with the given cameras.
            broadcast: Whether to broadcast new images.
            write_to: Write all images to this directory.
        """
        Module.__init__(self, *args, **kwargs)

        # store
        self._alt_range = tuple(alt_range)
        self._num_alt = num_alt
        self._az_range = tuple(az_range)
        self._num_az = num_az
        self._dec_range = dec_range
        self._min_moon_dist = min_moon_dist
        self._finish = 1. - finish / 100.
        self._randomize = randomize
        self._two_way = two_way
        self._exp_time = exp_time
        self._acquisition = acquisition
        self._telescope = telescope
        self._cameras = [] if cameras is None else cameras
        self._acquire = acquire
        self._broadcast = broadcast
        self._write_to = write_to

        # if Az range is [0, 360], we got north double, so remove one step
        if self._az_range == (0., 360.):
            self._az_range = (0., 360. - 360. / self._num_az)

        # add thread func
        self.add_thread_func(self._run_thread, False)

    def start(self, *args, **kwargs):
        """Starts a service."""
        pass

    def stop(self, *args, **kwargs):
        """Stops a service."""
        pass

    def is_running(self, *args, **kwargs) -> bool:
        """Whether a service is running."""
        return True

    def _run_thread(self):
        """Run a pointing series."""

        # create grid
        grid = []
        for az in np.linspace(self._az_range[0], self._az_range[1], self._num_az):
            for alt in np.linspace(self._alt_range[0], self._alt_range[1], self._num_alt):
                grid.append((alt, az))

        # randomize?
        if self._randomize:
            random.shuffle(grid)
        else:
            # two way? only if not randomizing.
            if self._two_way:
                grid += reversed(grid)

        # to dataframe
        grid = pd.DataFrame(grid).set_index(['alt', 'az'])

        # get telescope and acqusition or camera modules
        telescope: ITelescope = self.proxy(self._telescope, ITelescope)
        acquisition: IAcquisition = self.proxy(self._acquisition, IAcquisition) if self._acquisition else None
        cameras = [self.proxy(c, ICamera) for c in self._cameras]
        
        # set cameras
        for cam in cameras:
            if isinstance(cam, ICameraExposureTime):
                cam.set_exposure_time(self._exp_time)
            if isinstance(cam, ICameraBinning):
                cam.set_binning(2, 2)
            if isinstance(cam, IImageType):
                cam.set_image_type(ImageType.OBJECT)

        # loop until finished
        while not self.closing.is_set():
            # get all entries without offset measurements
            todo = list(grid[~grid['done']].index)
            if len(todo) / len(grid) < self._finish:
                log.info('Finished.')
                break
            log.info('Grid points left to do: %d', len(todo))

            # get moon
            moon = self.observer.moon_altaz(Time.now())

            # try to find a good point
            while True:
                # aborted or not running?
                if self.closing.is_set():
                    return

                # pick a random index and remove from list
                alt, az = random.sample(todo, 1)[0]
                todo.remove((alt, az))
                altaz = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame='altaz', obstime=Time.now(),
                                 location=self.observer.location)

                # get RA/Dec
                radec = altaz.icrs

                # moon far enough away?
                if altaz.separation(moon).degree >= self._min_moon_dist:
                    # yep, are we in declination range?
                    if self._dec_range[0] <= radec.dec.degree < self._dec_range[1]:
                        # yep, break here, we found our target
                        break

                # to do list empty?
                if len(todo) == 0:
                    # could not find a grid point
                    log.info('Could not find a suitable grid point, resetting todo list for next entry...')
                    todo = list(grid.index)
                    continue

            # log finding
            log.info('Picked grid point at Alt=%.2f, Az=%.2f (%s).', alt, az, radec.to_string('hmsdms'))

            # acquire target and process result
            try:
                # move telescope
                telescope.move_radec(float(radec.ra.degree), float(radec.dec.degree)).wait()

                # acquire target
                acq = acquisition.acquire_target(self._exp_time).wait()

                #  process result
                if acq is not None:
                    self._process_acquisition(**acq)

            except (ValueError, InvocationException):
                log.info('Could not acquire target.')
                continue

            # finished
            grid.loc[alt, az] = True

        # finished
        if self.closing.is_set():
            log.info('Pointing series aborted.')
        else:
            log.info('Pointing series finished.')

    def _process_acquisition(self, datetime: str, ra: float, dec: float, alt: float, az: float,
                             off_ra: float = None, off_dec: float = None, off_alt: float = None, off_az: float = None):
        """Process the result of the acquisition. Either ra_off/dec_off or alt_off/az_off must be given.

        Args:
            datetime: Date and time of observation.
            ra: Right ascension without offsets at destination.
            dec: Declination without offsets at destination.
            alt: Altitude without offsets at destination.
            az: Azimuth without offsets at destination.
            off_ra: Found RA offset.
            off_dec: Found Dec offset.
            off_alt: Found Alt offset.
            off_az: Found Az offset.
        """
        pass


__all__ = ['PointingSeries']
