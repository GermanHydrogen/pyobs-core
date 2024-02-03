from unittest.mock import Mock

import numpy as np
import pytest
from astropy.table import Table
from photutils.psf import EPSFStar
from pytest_mock import MockerFixture

from pyobs.images import Image
from pyobs.images.meta import PixelOffsets
from pyobs.images.processors.offsets import NStarOffsets
from pyobs.images.processors.offsets.nstar._gaussian_fitter import GaussianFitter


@pytest.mark.asyncio
async def test_reset() -> None:
    offsets = NStarOffsets()
    offsets.ref_boxes = [np.ones((2, 2))]
    await offsets.reset()

    assert len(offsets.ref_boxes) == 0


def test_get_box_size() -> None:
    assert NStarOffsets._get_box_size(4, 2) == 8

    assert NStarOffsets._get_box_size(4, None) == 20


def test_calculate_offsets_invalid_data() -> None:
    image = Image()

    offsets = NStarOffsets()

    assert offsets._calculate_offsets(image) == (None, None)


def test_calculate_offsets() -> None:
    data = np.array([
        [
            GaussianFitter._gauss2d([x, y], 1, 2, 10, 10, 1, 1) for x in range(21)
        ] for y in range(21)
    ])

    offsets = NStarOffsets()
    offsets.ref_boxes = [EPSFStar(data[7:14, 7:14], origin=(8, 8))]

    result = offsets._calculate_offsets(Image(data=data))
    np.testing.assert_almost_equal(result, (-1.0, -1.0), 2)


@pytest.mark.asyncio
async def test_call_init() -> None:
    image = Image()
    offsets = NStarOffsets()

    boxes = [EPSFStar(np.ones((10, 10)), origin=(8, 8))]
    offsets._box_generator = Mock(return_value=boxes)

    result = await offsets(image)
    assert result.get_meta(PixelOffsets).dx == 0.0
    assert result.get_meta(PixelOffsets).dy == 0.0

    assert offsets.ref_boxes == boxes


@pytest.mark.asyncio
async def test_call_invalid_init() -> None:
    image = Image()
    image.set_meta(PixelOffsets(1.0, 1.0))
    offsets = NStarOffsets()
    offsets._box_generator = Mock(side_effect=ValueError)

    result = await offsets(image)

    assert not result.has_meta(PixelOffsets)

    assert offsets.ref_boxes == []


@pytest.mark.asyncio
async def test_call(mocker: MockerFixture) -> None:
    image = Image()
    offsets = NStarOffsets()
    offsets.ref_boxes = [EPSFStar(np.ones((10, 10)), origin=(8, 8))]
    mocker.patch.object(offsets, "_calculate_offsets", return_value=(1.0, -1.0))

    result = await offsets(image)
    assert result.get_meta(PixelOffsets).dx == 1.0
    assert result.get_meta(PixelOffsets).dy == -1.0

