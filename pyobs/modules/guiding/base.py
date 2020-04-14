from typing import Union
import logging
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy.wcs import WCS
import astropy.units as u
from pyobs.utils.time import Time

from pyobs.interfaces import IAutoGuiding, IFitsHeaderProvider, ITelescope, IRaDecOffsets, IAltAzOffsets, ICamera
from pyobs import PyObsModule, get_object
from pyobs.mixins import TableStorageMixin
from pyobs.utils.guiding.base import BaseGuidingOffset
from pyobs.utils.images import Image


log = logging.getLogger(__name__)


class BaseGuiding(PyObsModule, TableStorageMixin, IAutoGuiding, IFitsHeaderProvider):
    def __init__(self, camera: Union[str, ICamera], telescope: Union[str, ITelescope],
                 offsets: Union[dict, BaseGuidingOffset], max_offset: float = 30, max_exposure_time: float = None,
                 max_interval: float = 600, separation_reset: float = None, pid: bool = False, log_file: str = None,
                 *args, **kwargs):
        """Initializes a new science frame auto guiding system.

        Args:
            telescope: Telescope to use.
            offsets: Auto-guider to use
            max_offset: Max offset in arcsec to move.
            max_exposure_time: Maximum exposure time in sec for images to analyse.
            max_interval: Maximum interval in sec between to consecutive images to guide.
            separation_reset: Min separation in arcsec between two consecutive images that triggers a reset.
            pid: Whether to use a PID for guiding.
            log_file: Name of file to write log to.
        """
        PyObsModule.__init__(self, *args, **kwargs)

        # store
        self._camera = camera
        self._telescope = telescope
        self._enabled = False
        self._max_offset = max_offset
        self._max_exposure_time = max_exposure_time
        self._max_interval = max_interval
        self._separation_reset = separation_reset
        self._pid = pid
        self._loop_closed = False

        # headers of last and of reference image
        self._last_header = None
        self._ref_header = None

        # create auto-guiding system
        self._guiding_offset: BaseGuidingOffset = get_object(offsets, BaseGuidingOffset)

        # columns for storage
        storage_columns = {
            'datetime': str,
            'ra': float,
            'dec': float,
            'alt': float,
            'az': float,
            'off_ra': float,
            'off_dec': float,
            'off_alt': float,
            'off_az': float
        }

        # init table storage and load measurements
        TableStorageMixin.__init__(self, filename=log_file, columns=storage_columns, reload_always=True)

    def open(self):
        """Open module."""
        PyObsModule.open(self)

        # check telescope
        try:
            self.proxy(self._telescope, ITelescope)
        except ValueError:
            log.warning('Given telescope does not exist or is not of correct type at the moment.')

        # check camera
        try:
            self.proxy(self._camera, ICamera)
        except ValueError:
            log.warning('Given camera does not exist or is not of correct type at the moment.')

    def start(self, *args, **kwargs):
        """Starts/resets auto-guiding."""
        log.info('Start auto-guiding...')
        self._reset_guiding(enabled=True)

    def stop(self, *args, **kwargs):
        """Stops auto-guiding."""
        log.info('Stopping autp-guiding...')
        self._reset_guiding()

    def is_running(self, *args, **kwargs) -> bool:
        """Whether auto-guiding is running.

        Returns:
            Auto-guiding is running.
        """
        return self._enabled

    def get_fits_headers(self, namespaces: list = None, *args, **kwargs) -> dict:
        """Returns FITS header for the current status of this module.

        Args:
            namespaces: If given, only return FITS headers for the given namespaces.

        Returns:
            Dictionary containing FITS headers.
        """

        # state
        state = 'GUIDING_CLOSED_LOOP' if self._loop_closed else 'GUIDING_OPEN_LOOP'

        # return header
        return {
            'AGSTATE': state
        }

    def _reset_guiding(self, enabled: bool = True, image: Union[Image, None] = None):
        """Reset guiding.

        Args:
            image: If given, new reference image.
        """
        self._enabled = enabled
        self._loop_closed = False
        self._guiding_offset.set_reference_image(image)
        self._ref_header = None if image is None else image.header

    def _process_image(self, image: Image):
        """Processes a single image and offsets telescope.

        Args:
            image: Image to process.
        """

        # we only accept OBJECT images
        if image.header['IMAGETYP'] != 'object':
            return

        # reference header?
        if self._ref_header is None:
            log.info('Setting new reference image...')
            self._reset_guiding(image=image)

        # check RA/Dec in header and separation
        c1 = SkyCoord(ra=image.header['TEL-RA'] * u.deg, dec=image.header['TEL-DEC'] * u.deg, frame='icrs')
        c2 = SkyCoord(ra=self._ref_header['TEL-RA'] * u.deg, dec=self._ref_header['TEL-DEC'] * u.deg, frame='icrs')
        separation = c1.separation(c2).deg
        if self._separation_reset is not None and separation * 3600. > self._separation_reset:
            log.warning('Nominal position of reference and new image differ by %.2f", resetting reference...',
                            separation * 3600.)
            self._reset_guiding(image=image)
            return

        # check filter
        if 'FILTER' in image.header and 'FILTER' in self._ref_header and \
                image.header['FILTER'] != self._ref_header['FILTER']:
            log.warning('The filter has been changed since the last exposure, resetting reference...')
            self._reset_guiding(image=image)
            return

        # get time
        date_obs = Time(image.header['DATE-OBS'])

        # check times and focus
        if self._last_header is not None:
            # check times
            t0 = Time(self._last_header['DATE-OBS'])
            if (date_obs - t0).sec > self._max_interval:
                log.warning('Time between current and last image is too large, resetting reference...')
                self._reset_guiding(image=image)
                return

            # check focus
            if abs(image.header['TEL-FOCU'] - self._last_header['TEL-FOCU']) > 0.05:
                log.warning('Focus difference between current and last image is too large, resetting reference...')
                self._reset_guiding(image=image)
                return

        # remember header
        self._last_header = image.header

        # get offset
        dx, dy = self._guiding_offset.find_pixel_offset(image)

        if dx is None or dy is None:
            log.error('Could not correlate image with reference.')
            return
        else:
            log.info('Found pixel shift of dx=%.2f, dy=%.2f.', dx, dy)

        # get reference pixel
        cx, cy = image.header['CRPIX1'], image.header['CRPIX1']

        # get WCS and RA/DEC for pixel and pixel + dx/dy
        w = WCS(image.header)
        lon, lat = w.all_pix2world(cx, cy, 0)
        radec1 = SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame='icrs', obstime=date_obs, location=self.location)
        lon, lat = w.all_pix2world(cx + dx, cy + dy, 0)
        radec2 = SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame='icrs', obstime=date_obs, location=self.location)

        # get telescope
        try:
            telescope: ITelescope = self.proxy(self._telescope, ITelescope)
        except ValueError:
            log.error('Given telescope does not exist or is not of correct type.')
            return

        # get current position
        cur_ra, cur_dec = telescope.get_radec().wait()

        # calculate offsets
        dra = (radec2.ra.degree - radec1.ra.degree) * np.cos(np.radians(cur_dec))
        ddec = radec2.dec.degree - radec1.dec.degree
        log.info('Transformed to RA/Dec shift of dRA=%.2f", dDec=%.2f".', dra * 3600., ddec * 3600.)

        # too large?
        if abs(dra * 3600.) > self._max_offset or abs(ddec * 3600.) > self._max_offset:
            log.warning('Shift too large, skipping auto-guiding for now...')
            return

        # exposure time too large
        if self._max_exposure_time is not None and image.header['EXPTIME'] > self._max_exposure_time:
            log.warning('Exposure time too large, skipping auto-guiding for now...')
            return

        # push offset into PID
        #if self._pid:
        #    dra = self._pid_ra.update(dra)
        #    ddec = self._pid_dec.update(ddec)
        #    log.info('PID results in RA/Dec shift of dRA=%.2f", dDec=%.2f.', dra * 3600., ddec * 3600.)

        # is telescope on an equitorial mount?
        if isinstance(telescope, IRaDecOffsets):
            # get current offset
            cur_dra, cur_ddec = telescope.get_radec_offsets().wait()

            # move offset
            log.info('Offsetting telescope...')
            telescope.set_radec_offsets(float(cur_dra + dra), float(cur_ddec + ddec)).wait()
            log.info('Finished image.')
            self._loop_closed = True

        elif isinstance(telescope, IAltAzOffsets):
            # transform both to Alt/AZ
            altaz1 = radec1.transform_to(AltAz)
            altaz2 = radec2.transform_to(AltAz)

            # calculate offsets
            dalt = altaz2.alt.degree - altaz1.alt.degree
            daz = altaz2.az.degree - altaz1.az.degree
            log.info('Transformed to Alt/Az shift of dalt=%.2f", daz=%.2f.', dalt * 3600., daz * 3600.)

            # get current offset
            cur_dalt, cur_daz = telescope.get_altaz_offsets().wait()

            # move offset
            log.info('Offsetting telescope...')
            telescope.set_altaz_offsets(float(cur_dalt + dalt), float(cur_daz + daz)).wait()
            log.info('Finished image.')
            self._loop_closed = True

        else:
            log.warning('Telescope has neither altaz nor equitorial mount. No idea how to move it...')


__all__ = ['BaseGuiding']
