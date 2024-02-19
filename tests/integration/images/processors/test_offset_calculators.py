from typing import Tuple

import numpy as np
import pytest
from astropy.io import fits

from pyobs.images import Image
from pyobs.images.meta import PixelOffsets
from pyobs.images.processors.detection import DaophotSourceDetection
from pyobs.images.processors.offsets import *


@pytest.fixture()
def test_images() -> Tuple[Image, Image]:
    image = Image.from_file("new-image.fits")
    data = image.data

    ref_data = np.pad(data, (1, 0), mode='constant')
    ref_data = ref_data[:-1]  # Prune at the end to keep the size the same

    return image, Image(ref_data)


async def test_projected(test_images: Tuple[Image, Image]) -> None:
    ref_image, test_image = test_images
    calculator = ProjectedOffsets()

    await calculator(ref_image)
    result = await calculator(test_image)

    pixel_offset = result.get_meta(PixelOffsets)
    np.testing.assert_almost_equal(pixel_offset.dy, 1.0, 0)
    np.testing.assert_almost_equal(pixel_offset.dx, 0.0, 0)


async def test_astrometry(test_images: Tuple[Image, Image]) -> None:
    """
    Test image includes header from astrometry.net
    """
    ref_image, test_image = test_images
    calculator = AstrometryOffsets()

    result = await calculator(ref_image)

    pixel_offset = result.get_meta(PixelOffsets)
    np.testing.assert_almost_equal(pixel_offset.dy, 5603.603992688688)
    np.testing.assert_almost_equal(pixel_offset.dx, -1176.2827278126224)


async def test_brightest_star(test_images: Tuple[Image, Image]) -> None:
    ref_image, test_image = test_images

    with fits.open("wcs.fits") as wcs_file:
        wcs_header = wcs_file[0].header

    for key, value in wcs_header.items():
        test_image.header[key] = value

    calculator = BrightestStarOffsets()

    source_detector = DaophotSourceDetection()
    test_image = await source_detector(test_image)

    result = await calculator(test_image)

    pixel_offset = result.get_meta(PixelOffsets)
    np.testing.assert_almost_equal(pixel_offset.dy, -300.07660569777755)
    np.testing.assert_almost_equal(pixel_offset.dx, 307.23826245427415)