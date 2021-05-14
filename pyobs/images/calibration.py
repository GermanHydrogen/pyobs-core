from typing import List, Type, Union
import numpy as np
from astropy.stats import sigma_clip
import astropy.units as u
import logging

from pyobs.interfaces import ICamera
from pyobs.utils.time import Time
from .image import Image
from pyobs.utils.archive import FrameInfo
from pyobs.utils.enums import ImageType

log = logging.getLogger(__name__)


class CalibrationImage(Image):
    @staticmethod
    def combine(images: List[Image], method: Image.CombineMethod = Image.CombineMethod.MEAN):
        """Combines images into a single one.

        Args:
            images: Images to combine.
            method: Method to use for combination.

        Returns:
            Combined image.
        """

        # collect data
        data = [img.data for img in images]

        # create new image
        img = Image()

        # average
        if method == Image.CombineMethod.MEAN:
            img.data = np.mean(data, axis=0)
        elif method == Image.CombineMethod.MEDIAN:
            img.data = np.median(data, axis=0)
        elif method == Image.CombineMethod.SIGMA:
            tmp = sigma_clip(data, axis=0)
            img.data = np.mean(tmp, axis=0)
        else:
            raise ValueError('Unknown combine method.')

        # header
        img.header = images[0].header.copy()

        # add history
        for i, src in enumerate(images, 1):
            basename = src.header['FNAME'].replace('.fits.fz', '').replace('.fits', '')
            img.header['L1AVG%03d' % i] = (basename, 'Image used for average')
        img.header['RLEVEL'] = (1, 'Reduction level')

        # finished
        return img

    @staticmethod
    def find_master(image_type: ImageType , archive: 'Archive', time: Time, instrument: str,
                    binning: str, filter_name: str = None, max_days: float = 30.) -> Union[None, FrameInfo]:
        """Find and download master calibration frame.

        Args:
            image_type: Image type.
            archive: Archive to use for downloading frames.
            time: Time to search at.
            instrument: Instrument to use.
            binning: Used binning.
            filter_name: Used filter.
            max_days: Maximum number of days from DATE-OBS to find frames.

        Returns:
            FrameInfo for master calibration frame or None.
        """

        # find reduced frames from +- N days
        log.info('Searching for %s %s master calibration frames%s from instrument %s.',
                 binning, image_type.value, '' if filter_name is None else ' in ' + filter_name, instrument)
        infos = archive.list_frames(start=time - max_days * u.day, end=time + max_days * u.day,
                                    instrument=instrument, image_type=image_type, binning=binning,
                                    filter_name=filter_name, rlevel=1)

        # found any?
        if len(infos) == 0:
            log.warning('Could not find any matching %s calibration frames.', image_type.value)
            return None
        else:
            # sort by diff to time and take first
            s = sorted(infos, key=lambda i: abs((i.dateobs - time).sec))
            info = s[0]
            log.info('Found %s frame %s.', image_type.name, info.filename)

            # return FrameInfo
            return info


__all__ = ['CalibrationImage']
