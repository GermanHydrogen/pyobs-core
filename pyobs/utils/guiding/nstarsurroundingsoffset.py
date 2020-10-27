import logging
import numpy as np
from scipy import signal, interpolate, optimize
from astropy.nddata import NDData
from astropy.table import Table, Column
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin
import re
import photutils

from pyobs.utils.images import Image
from pyobs.utils.photometry.sep import SepPhotometry
from pyobs.utils.pid import PID
from .base import BaseGuidingOffset

import matplotlib.pyplot as plt
log = logging.getLogger(__name__)


class CorrelationMaxCloseToBorderError(Exception):
    pass


class NStarSurroundingsOffset(BaseGuidingOffset):
    """An auto-guiding system based on comparing 2D images of the surroundings of variable number of stars."""

    MIN_REQUIRED_SOURCES_IN_IMAGE = 5

    def __init__(
        self, N_stars=1, max_expected_offset_in_arcsec=1e-3 * 3600, *args, **kwargs
    ):
        """Initializes a new auto guiding system."""
        print("Initializing NStarSurroundingsOffset")
        self.N_stars = N_stars
        self.max_expected_offset_in_arcsec = max_expected_offset_in_arcsec
        self._ref_box_dimensions = None
        self._ref_boxed_images = None
        self.star_box_size = None
        self._pid_ra = None
        self._pid_dec = None

    def reset(self):
        """Resets guiding."""
        log.info("Reset autp-guiding.")
        self._ref_box_dimensions = None
        self._ref_boxed_images = None

    def find_pixel_offset(
        self,
        image: Image,
    ) -> (float, float):
        """Processes an image and return x/y pixel offset to reference.

        Args:
            image: Image to process.

        Returns:
            x/y pixel offset to reference.

        Raises:
            ValueError if offset could not be found.
        """

        # no reference image?
        if self._ref_box_dimensions is None or self._ref_boxed_images is None:
            log.info("Initialising auto-guiding with new image...")
            self.star_box_size = self.get_star_box_size_from_max_expected_offset(
                self.max_expected_offset_in_arcsec, image.pixel_scale
            )

            # initialize reference image information: dimensions & position of boxes, box images
            (
                self._ref_box_dimensions,
                self._ref_boxed_images,
            ) = self._create_star_boxes_from_ref_image(image)
            self._init_pid()
            return 0, 0

        # process it
        log.info("Perform auto-guiding on new image...")
        dx, dy = self.calculate_offset(image)
        return dx, dy

    @staticmethod
    def get_star_box_size_from_max_expected_offset(
        max_expected_offset_in_arcsec, pixel_scale
    ):
        # multiply by 4 to give enough space for fit of correlation around the peak on all sides
        star_box_size = int(
            4 * max_expected_offset_in_arcsec / pixel_scale if pixel_scale else 20
        )
        return star_box_size

    def _create_star_boxes_from_ref_image(self, image: Image) -> (list, list):
        sep = SepPhotometry()
        sources: Table = sep.find_stars(image)
        sources = self.convert_from_fits_to_numpy_index_convention(sources)

        # filter sources
        sources = self.remove_sources_close_to_border(
            sources, image.data.shape, self.star_box_size // 2 + 1
        )
        sources = self.remove_bad_sources(sources)
        self.check_if_enough_sources_in_image(sources, self.N_stars)
        selected_sources = self.select_top_N_brightest_sources(self.N_stars, sources)

        # find positions & dimensions of boxes around the stars, and the corresponding box images
        box_dimensions, boxed_images = [], []
        for i_selected_source, _ in enumerate(selected_sources):
            # make astropy table with only the selected source
            single_source_catalog = Table(rows=selected_sources[i_selected_source])
            stars = photutils.psf.extract_stars(
                NDData(image.data), single_source_catalog, size=self.star_box_size
            )
            boxed_star_image = stars.all_stars[0].data

            box_dimensions.append(
                [
                    stars.all_stars[0].origin[1],
                    stars.all_stars[0].origin[1] + boxed_star_image.shape[0],
                    stars.all_stars[0].origin[0],
                    stars.all_stars[0].origin[0] + boxed_star_image.shape[1],
                ]
            )

            boxed_star_image = image.data[
                box_dimensions[-1][0] : box_dimensions[-1][1],
                box_dimensions[-1][2] : box_dimensions[-1][3],
            ]

            boxed_images.append(boxed_star_image)

        return box_dimensions, boxed_images

    @staticmethod
    def convert_from_fits_to_numpy_index_convention(sources: Table) -> Table:
        sources["x"] -= 1
        sources["y"] -= 1
        sources["xmin"] -= 1
        sources["xmax"] -= 1
        sources["ymin"] -= 1
        sources["ymax"] -= 1
        sources["xpeak"] -= 1
        sources["ypeak"] -= 1
        return sources

    def remove_sources_close_to_border(
        self, sources: Table, image_shape: tuple, min_distance_from_border_in_pixels
    ) -> Table:
        """Remove table rows from sources when closer than min_distance_from_border_in_pixels from border of image."""
        width, height = image_shape

        def min_distance_from_border(source):
            # minimum across x and y of distances to border
            return np.min(
                np.array(
                    (
                        (width / 2 - np.abs(source["y"] - width / 2)),
                        (height / 2 - np.abs(source["x"] - height / 2)),
                    )
                ),
                axis=0,
            )

        sources.add_column(
            Column(
                name="min_distance_from_border",
                data=min_distance_from_border(sources),
            )
        )
        sources.sort("min_distance_from_border")

        sources_result = sources[
            np.where(
                sources["min_distance_from_border"]
                > min_distance_from_border_in_pixels
            )
        ]
        return sources_result

    def remove_bad_sources(
        self,
        sources: Table,
        MAX_ELLIPTICITY=0.4,
        MIN_FACTOR_ABOVE_LOCAL_BACKGROUND: float = 1.5,
    ) -> Table:

        # remove large and small sources
        tnpix_median = np.median(sources["tnpix"])
        tnpix_std = np.std(sources["tnpix"])
        sources.sort("tnpix")
        sources = sources[
            np.where(
                np.logical_and(
                    tnpix_median - tnpix_std / 2 <= sources["tnpix"],
                    sources["tnpix"] <= tnpix_median + 2 * tnpix_std,
                )
            )
        ]

        # remove highly elliptic sources
        sources.sort("ellipticity")
        sources = sources[np.where(sources["ellipticity"] <= MAX_ELLIPTICITY)]

        # remove sources with background <= 0
        sources = sources[np.where(sources["background"] > 0)]
        # remove sources with low contrast to background
        sources = sources[
            np.where(
                (sources["peak"] + sources["background"]) / sources["background"]
                > MIN_FACTOR_ABOVE_LOCAL_BACKGROUND
            )
        ]
        return sources

    def select_top_N_brightest_sources(self, N_stars: int, sources: Table):
        sources.sort("flux")
        sources.reverse()
        if 0 < N_stars < len(sources):
            sources = sources[:N_stars]
        return sources

    def check_if_enough_sources_in_image(self, sources: Table, N_stars=0):
        n_required_sources = max(self.MIN_REQUIRED_SOURCES_IN_IMAGE, N_stars)
        if len(sources) < n_required_sources:
            raise ValueError(
                f"Only {len(sources)} source(s) in image, but at least {n_required_sources} required."
            )

    def calculate_offset(
        self,
        current_image: Image,
    ) -> tuple:

        # calculate offset for each star
        offsets = []
        for ref_box_dimension, ref_boxed_image in zip(
            self._ref_box_dimensions, self._ref_boxed_images
        ):
            box_ymin, box_ymax, box_xmin, box_xmax = ref_box_dimension
            current_boxed_image = current_image.data[
                box_ymin:box_ymax, box_xmin:box_xmax
            ]

            corr = signal.correlate2d(
                current_boxed_image, ref_boxed_image, mode="same", boundary="wrap"
            )

            try:
                offset = self.calculate_offset_from_2d_correlation(corr)
                offsets.append(offset)
            except Exception as e:
                log.info(f"Exception '{e}' caught. Ignoring this star.")
                pass

        if len(offsets) == 0:
            log.info(
                f"All {self.N_stars} fits on boxed star correlations failed."
            )
            return None, None
        offsets = np.array(offsets)

        offset = np.mean(offsets[:, 0]), np.mean(offsets[:, 1])

        return offset

    def gauss2d(self, x, a, b, x0, y0, sigma_x, sigma_y):
        return a + b * np.exp(
            -((x[0] - x0) ** 2) / (2 * sigma_x ** 2)
            - (x[1] - y0) ** 2 / (2 * sigma_y ** 2)
        )

    def calculate_offset_from_2d_correlation(self, corr):
        """Fit 2d correlation data with a 2d gaussion + constant offset.
        raise CorrelationMaxCloseToBorderError if the correlation maximum is not well separated from border."""
        # calc positions corresponding to the values in the correlation
        xs = np.arange(-corr.shape[0] / 2, corr.shape[0] / 2) + 0.5
        ys = np.arange(-corr.shape[1] / 2, corr.shape[1] / 2) + 0.5
        X, Y = np.meshgrid(xs, ys)

        # format data as needed by R^2 -> R curve_fit
        xdata = np.vstack((X.ravel(), Y.ravel()))
        ydata = corr.ravel()

        # initial parameter values
        a = np.min(corr)
        b = np.max(corr) - a
        # use max pixel as initial value for x0, y0
        max_index = np.array(np.unravel_index(np.argmax(corr), corr.shape))
        x0, y0 = X[tuple(max_index)], Y[tuple(max_index)]
        self.check_if_correlation_max_is_close_to_border(corr)

        sigma_x, sigma_y = 3, 3  # TODO: dependence on binning
        p0 = [a, b, x0, y0, sigma_x, sigma_y]
        bounds = (
            [-np.inf, -np.inf, x0 - sigma_x, y0 - sigma_y, 0, 0],
            [np.inf, np.inf, x0 + sigma_x, y0 + sigma_y, np.inf, np.inf],
        )
        # only use data that clearly belong to peak to avoid border effects
        mask_value_above_background = ydata > -1e5  # a + .1*b
        mask_circle_around_peak = (X.ravel() - x0) ** 2 + (Y.ravel() - y0) ** 2 < 4 * (
            sigma_x ** 2 + sigma_y ** 2
        ) / 2
        mask = np.logical_and(mask_value_above_background, mask_circle_around_peak)
        ydata_restricted = ydata[mask]
        xdata_restricted = xdata[:, mask]

        try:
            popt, pcov = optimize.curve_fit(
                self.gauss2d,
                xdata_restricted,
                ydata_restricted,
                p0,
                bounds=bounds,
                maxfev=int(1e5),
                ftol=1e-12,
                # verbose=2
            )
        except Exception as e:
            # if fit fails return max pixel
            log.info(e)
            log.info("Returning pixel position with maximal value in correlation.")
            offset = np.unravel_index(np.argmax(corr), corr.shape)
            return offset

        MEDIAN_SQUARED_RELATIVE_RESIDUE_THRESHOLD = 1e-2
        fit_ydata_restricted = self.gauss2d(xdata_restricted, *popt)
        square_rel_res = np.square(
            (fit_ydata_restricted - ydata_restricted) / fit_ydata_restricted
        )
        median_squared_rel_res = np.median(np.square(square_rel_res))

        if median_squared_rel_res > MEDIAN_SQUARED_RELATIVE_RESIDUE_THRESHOLD:
            raise Exception(
                f"Bad fit with median squared relative residue = {median_squared_rel_res}"
                f" vs allowed value of {MEDIAN_SQUARED_RELATIVE_RESIDUE_THRESHOLD}"
            )

        return (popt[2], popt[3])

    def check_if_correlation_max_is_close_to_border(self, corr):
        corr_size = corr.shape[0]

        xs = np.arange(-corr.shape[0] / 2, corr.shape[0] / 2) + 0.5
        ys = np.arange(-corr.shape[1] / 2, corr.shape[1] / 2) + 0.5

        X, Y = np.meshgrid(xs, ys)

        max_index = np.array(np.unravel_index(np.argmax(corr), corr.shape))
        x0, y0 = X[tuple(max_index)], Y[tuple(max_index)]

        if (
            x0 < -corr_size / 4
            or x0 > corr_size / 4
            or y0 < -corr_size / 4
            or y0 > corr_size / 4
        ):
            raise CorrelationMaxCloseToBorderError(
                "Maximum of correlation is outside center half of axes. "
                "This means that either the given image data is bad, or the offset is larger than expected."
            )

    def _init_pid(self):
        # init pids
        Kp = 0.2
        Ki = 0.16
        Kd = 0.83

        # reset
        self._pid_ra = PID(Kp, Ki, Kd)
        self._pid_dec = PID(Kp, Ki, Kd)


__all__ = ["NStarSurroundingsOffset"]
