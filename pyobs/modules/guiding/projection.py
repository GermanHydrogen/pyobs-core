import logging
import threading
from typing import Union
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy.io import fits
from astropy.time import Time
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin
from astropy.wcs import WCS
import re

from pyobs import PyObsModule
from pyobs.events import NewImageEvent
from pyobs.interfaces import ITelescope, IAutoGuiding, IStoppable, IEquitorialMount, IAltAzMount
from pyobs.utils.pid import PID


log = logging.getLogger(__name__)


class AutoGuidingProjection(PyObsModule, IAutoGuiding, IStoppable):
    """An auto-guiding system based on comparing collapsed images along the x&y axes with a reference image."""

    def __init__(self, telescope: Union[str, ITelescope], max_offset: float = 10, max_exposure_time: float = 20,
                 max_interval: float = 30, separation_reset: float = 10, new_images_channel: str = 'new_images',
                 pid: bool = False, *args, **kwargs):
        """Initializes a new auto guiding system.

        Args:
            telescope: Telescope to use.
            max_offset: Max offset in arcsec to move.
            max_exposure_time: Maximum exposure time in sec for images to analyse.
            max_interval: Maximum interval in sec between to consecutive images to guide.
            separation_reset: Min separation in arcsec between two consecutive images that triggers a reset.
            new_images_channel: Channel for receiving new images.
            pid: Whether to use a PID for guiding.
        """
        PyObsModule.__init__(self, thread_funcs=self._auto_guiding, *args, **kwargs)

        # store
        self._telescope = telescope
        self._max_offset = max_offset
        self._max_exposure_time = max_exposure_time
        self._max_interval = max_interval
        self._separation_reset = separation_reset
        self._new_images_channel = new_images_channel
        self._pid = pid

        # variables
        self._ref_image = None
        self._ref_header = None
        self._last_header = None
        self._next_image = None
        self._enabled = True
        self._pid_ra = None
        self._pid_dec = None
        self._lock = threading.Lock()

    def open(self):
        """Open module."""
        if not PyObsModule.open(self):
            return False

        # check telescope
        try:
            self.proxy(self._telescope, ITelescope)
        except ValueError:
            log.warning('Given telescope does not exist or is not of correct type at the moment.')

        # subscribe to channel with new images
        log.info('Subscribing to new image events...')
        self.comm.register_event(NewImageEvent, self.add_image)

    @staticmethod
    def _gaussian(pars, x):
        a = pars[0]
        x0 = pars[1]
        sigma = pars[2]
        return a * np.exp(-((x - x0) ** 2) / (2. * sigma ** 2))

    @staticmethod
    def _gaussian_fit(pars, y, x):
        err = y - AutoGuidingProjection._gaussian(pars, x)
        return (err * err).sum()

    @staticmethod
    def _correlate(data1, data2, fit_width=10):
        # do cross-correlation
        corr = np.correlate(data1, data2, "full")

        # find index of maximum
        i_max = np.argmax(corr)
        centre = i_max - data1.size + 1

        # cut window
        x = np.linspace(centre - fit_width, centre + fit_width, 2 * fit_width + 1)
        y = corr[i_max - fit_width:i_max + fit_width + 1]

        # moment calculation for initial guesses
        total = float(y.sum())
        m = (x * y).sum() / total
        m2 = (x * x * y).sum() / total - m**2

        # initial guess
        guesses = [np.max(y), m, m2]

        # perform fit
        result = fmin(AutoGuidingProjection._gaussian_fit, guesses, args=(y, x), disp=False)

        # sanity check and finish up
        shift = result[1]
        if shift < centre - fit_width or shift > centre + fit_width:
            return None
        return shift

    def _auto_guiding(self):
        """the thread function for processing the images"""

        # init pid
        self._init_pid()

        # run until closed
        while not self.closing.is_set():
            # get next image to process
            with self._lock:
                image = self._next_image

            # got one?
            if image:
                # process it
                if self._ref_image:
                    log.info('Perform auto-guiding on new image...')
                else:
                    log.info('Initialising auto-guiding with new image...')

                # process image
                try:
                    self._process_image(image)
                except:
                    log.exception('Something went wrong.')

                # image finished
                with self._lock:
                    self._next_image = None

            # wait for next image
            self.closing.wait(0.1)

    def _reset_guiding(self, sum_x, sum_y, hdr):
        # reset
        self._ref_image = (sum_x, sum_y)
        self._ref_header = hdr
        self._last_header = hdr
        self._init_pid()

    def _init_pid(self):
        # init pids
        Kp = 0.2
        Ki = 0.16
        Kd = 0.83

        # reset
        self._pid_ra = PID(Kp, Ki, Kd)
        self._pid_dec = PID(Kp, Ki, Kd)

    @staticmethod
    def _subtract_sky(data, frac=0.15, sbin=10):
        # find continuum for every of the sbin bins
        bins = np.zeros((sbin))
        binxs = np.zeros((sbin))
        x = list(range(len(data)))
        w1 = 0
        w2 = float(len(x)) / sbin
        for i in range(sbin):
            # sort data in range
            bindata = list(reversed(sorted(data[int(w1):int(w2)])))
            # calculate median and set wavelength
            bins[i] = np.median(bindata[int(-frac * len(bindata)):-1])
            binxs[i] = np.mean(x[int(w1):int(w2)])
            # reset ranges
            w1 = w2
            w2 += float(len(x)) / sbin
            # check for last bin
            if i == sbin - 1:
                w2 = len(x)

        # fit it
        w = np.where(~np.isnan(bins))
        ip = UnivariateSpline(binxs[w], bins[w])
        cont = ip(x)

        # return continuum
        return data - cont

    def _process_image(self, image: fits.PrimaryHDU):
        """Actually process an image.

        Args:
            image: Image to ptocess
        """

        # get image data and header
        data, hdr = image.data, image.header

        # trimsec
        if 'TRIMSEC' in hdr:
            m = re.match('\[([0-9]+):([0-9]+),([0-9]+):([0-9]+)\]', hdr['TRIMSEC'])
            x0, x1, y0, y1 = [int(f) for f in m.groups()]
            data = data[y0-1:y1, x0-1:x1]

        # collapse
        sum_x = np.nansum(data, 0)
        sum_y = np.nansum(data, 1)

        # sky subtraction
        sum_x = self._subtract_sky(sum_x)
        sum_y = self._subtract_sky(sum_y)

        # is this the new reference?
        if not self._ref_image:
            # yes, just store it
            self._reset_guiding(sum_x, sum_y, hdr)
            return

        # check RA/Dec in header and separation
        c1 = SkyCoord(ra=hdr['TEL-RA'] * u.deg, dec=hdr['TEL-DEC'] * u.deg, frame='icrs')
        c2 = SkyCoord(ra=self._ref_header['TEL-RA'] * u.deg, dec=self._ref_header['TEL-DEC'] * u.deg, frame='icrs')
        separation = c1.separation(c2).deg
        if separation * 3600. > self._separation_reset:
            log.warning('Nominal position of reference and new image differ by %.2f", resetting reference...',
                            separation * 3600.)
            self._reset_guiding(sum_x, sum_y, hdr)
            return

        # check filter
        if 'FILTER' in hdr and 'FILTER' in self._ref_header and hdr['FILTER'] != self._ref_header['FILTER']:
            log.warning('The filter has been changed since the last exposure, resetting reference...')
            self._reset_guiding(sum_x, sum_y, hdr)
            return

        # check times and focus
        if self._last_header is not None:
            # check times
            t = Time(hdr['DATE-OBS'])
            t0 = Time(self._last_header['DATE-OBS'])
            if (t - t0).sec > self._max_interval:
                log.warning('Time between current and last image is too large, resetting reference...')
                self._reset_guiding(sum_x, sum_y, hdr)
                return

            # check focus
            if abs(hdr['TEL-FOCU'] - self._last_header['TEL-FOCU']) > 0.05:
                log.warning('Focus difference between current and last image is too large, resetting reference...')
                self._reset_guiding(sum_x, sum_y, hdr)
                return

        # remember header
        self._last_header = hdr

        # find peaks
        dx = self._correlate(sum_x, self._ref_image[0])
        dy = self._correlate(sum_y, self._ref_image[1])
        if dx is None or dy is None:
            log.error('Could not correlate image with reference.')
            return
        else:
            log.info('Found pixel shift of dx=%.2f, dy=%.2f.', dx, dy)

        # get pixel in middle of image
        cx, cy = (np.array(data.shape) / 2.).astype(np.int)

        # get WCS and RA/DEC for pixel and pixel + dx/dy
        w = WCS(hdr)
        lon, lat = w.all_pix2world(cx, cy, 0)
        radec1 = SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame='icrs', obstime=t, location=self.location)
        lon, lat = w.all_pix2world(cx + dx, cy + dy, 0)
        radec2 = SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame='icrs', obstime=t, location=self.location)

        # calculate offsets
        dra = radec2.ra.degree - radec1.ra.degree
        ddec = radec2.dec.degree - radec1.dec.degree
        log.info('Transformed to RA/Dec shift of dRA=%.2f", dDec=%.2f".', dra * 3600., ddec * 3600.)

        # too large?
        max_offset = self._max_offset
        if abs(dra * 3600.) > max_offset or abs(ddec * 3600.) > max_offset:
            log.warning('Shift too large, skipping auto-guiding for now...')
            return

        # exposure time too large
        if hdr['EXPTIME'] > self._max_exposure_time:
            log.warning('Exposure time too large, skipping auto-guiding for now...')
            return

        # push offset into PID
        if self._pid:
            dra = self._pid_ra.update(dra)
            ddec = self._pid_dec.update(ddec)
            log.info('PID results in RA/Dec shift of dRA=%.2f", dDec=%.2f.', dra * 3600., ddec * 3600.)

        # get telescope
        telescope: ITelescope = self.comm[self._telescope]
        if not isinstance(telescope, ITelescope):
            log.error('Given telescope is not of type ITelescope, aborting.')
            return

        # is telescope on an equitorial mount?
        if isinstance(telescope, IEquitorialMount):
            # get current offset
            cur_dra, cur_ddec = telescope.get_radec_offsets().wait()

            # move offset
            log.info('Offsetting telescope...')
            telescope.set_radec_offsets(cur_dra + dra, cur_ddec + ddec).wait()
            log.info('Finished image.')

        elif isinstance(telescope, IAltAzMount):
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
            telescope.set_altaz_offsets(cur_dalt + dalt, cur_daz + daz).wait()
            log.info('Finished image.')

        else:
            log.warning('Telescope has neither altaz nor equitorial mount. No idea how to move it...')

    def start(self, *args, **kwargs) -> bool:
        """Starts/resets auto-guiding."""
        self._ref_image = None
        self._enabled = True

    def stop(self, *args, **kwargs):
        """Stops auto-guiding."""
        self._ref_image = None
        self._enabled = False

    def is_running(self, *args, **kwargs) -> bool:
        """Whether auto-guiding is running.

        Returns:
            Auto-guiding is running.
        """
        return self._enabled

    def add_image(self, event: NewImageEvent, sender: str, *args, **kwargs):
        """Processes an image asynchronously, returns immediately.

        Args:
            filename: Filename of image to process.
        """

        log.info('Received new image from %s.', sender)

        # if not enabled, just ignore
        if not self._enabled:
            return

        # download image
        try:
            with self.open_file(event.filename, 'rb') as f:
                tmp = fits.open(f, memmap=False)
                data = fits.PrimaryHDU(data=tmp[0].data, header=tmp[0].header)
                tmp.close()
        except FileNotFoundError:
            log.error('Could not download image.')
            return

        # we only accept OBJECT images
        if data.header['IMAGETYP'] != 'object':
            return

        # store filename as next image to process
        with self._lock:
            # do we have a filename in here already?
            if self._next_image:
                log.warning('Last image still being processed by auto-guiding, skipping new one.')
                return

            # store it
            self._next_image = data


__all__ = ['AutoGuidingProjection']
