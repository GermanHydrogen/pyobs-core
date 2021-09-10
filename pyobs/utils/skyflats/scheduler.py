import logging
import re
from typing import Dict, Union, Optional

from py_expression_eval import Parser
from astropy.time import Time, TimeDelta
import astropy.units as u
from astroplan import Observer
import operator

from .priorities import SkyflatPriorities


log = logging.getLogger(__name__)


class ExpTimeEval:
    """Exposure time evaluator for skyflats."""

    def __init__(self, observer: Observer, functions: Union[str, Dict[str, Union[str, Dict[str, str]]]]):
        """Initializes a new evaluator.

        Args:
            observer: Observer to use.
            functions: Dict of functions for the different filters/binnings.
                Three possible formats:
                1. Just a string with a function, e.g. 'exp(-0.9*(h+3.9))', completely ignoring binning and filter.
                2. Dictionary on filter or binning like
                   {'1x1': 'exp(-0.9*(h+3.9))'}
                   or
                   {'clear': 'exp(-0.9*(h+3.9))'}
                   If a binning is given, filters are ignored, and vice versa. Binnings need to be given as NxN.
                3. Nested dictionary with binning and filter like
                   {'1x1': {'clear': 'exp(-0.9*(h+3.9))'}}
                   In this structure, binning must be the first layer, followed by filter.
        """

        # init
        self._observer = observer
        self._time = None
        self._m = None
        self._b = None

        # get parser and init functions dict
        p = Parser()
        self._functions: Dict[(Optional[str], Optional[str])] = {}

        # so, what format is the functions dict?
        if isinstance(functions, str):
            # single function
            self._functions[None, None] = p.parse(functions)
            log.info('Found a single flatfield function for all binnings and filters.')

        else:
            # check, whether keys are binnings or filters
            is_binning = [re.match('[0-9]+[0-9]+', k) is not None for k in functions.keys()]
            if any(is_binning) and not all(is_binning):
                raise ValueError('Inconsistent configuration: first layer is neither all binnings nor all filters. ')

            # if all entries in is_binning are True, first layer is binnings
            if all(is_binning):
                # 1st level is binnings, is next level strings or another dict?
                is_str = [isinstance(f, str) for f in functions.values()]
                if any(is_str) and not all(is_str):
                    raise ValueError('Inconsistent configuration: second layer is neither all str nor all dicts.')

                # filters or not?
                if all(is_str):
                    # all strings, so we don't have filters
                    self._functions = {(b, None): p.parse(func) for b, func in functions.items()}

                else:
                    # need to go a level deeper
                    self._functions = {b: {f: p.parse(func) for f, func in tmp} for b, tmp in functions.items()}

            else:
                # 1st level is filters, second level must be strings!
                is_str = [isinstance(f, str) for f in functions.values()]
                if not all(is_str):
                    raise ValueError('Inconsistent configuration: second level must be functions.')

                # parse
                self._functions = {(None, f): p.parse(func) for f, func in functions.items()}

    def __call__(self, solalt: float, binning: int = None, filter_name: str = None) -> float:
        """Estimate exposure time for given filter

        Args:
            solalt: Solar altitude.
            binning: Used binning in X and Y.
            filter_name: Name of filter.

        Returns:
            Estimated exposure time.
        """

        # build binning string (if given)
        sbin = None if binning is None else '%dx%d' % (binning, binning)

        # get function and evaluate it
        exptime = self._functions[sbin, filter_name].evaluate({'h': solalt})

        # if binning is given, we return exptime directly, otherwise we scale with binning
        return exptime / binning**2 if binning is not None else exptime

    def init(self, time: Time):
        """Initialize object with the given time.

        Args:
            time: Start time for all further calculations.
        """

        # store time
        self._time = time

        # get sun now and in 10 minutes
        sun_now = self._observer.sun_altaz(time)
        sun_10min = self._observer.sun_altaz(time + TimeDelta(10 * u.minute))

        # get m, b for calculating sun_alt=m*time+b
        self._b = sun_now.alt.degree
        self._m = (sun_10min.alt.degree - self._b) / (10. * 60.)

    def exp_time(self, filter_name: str, binning: int, time_offset: float) -> float:
        """Estimates exposure time for a given filter and binning at a given time offset from the start time (see init).

        Args:
            filter_name: Name of filter
            binning: Used binning in X and Y
            time_offset: Offset in seconds from start time (see init)

        Returns:
            Estimated exposure time
        """
        return self(filter_name, binning, self._m * time_offset + self._b)

    def duration(self, filter_name: str, binning: int, count: int, start_time: float = 0, readout: float = 0) -> float:
        """Estimates the duration for a given amount of flats in the given filter and binning, starting at the given
        start time.

        Args:
            filter_name: Name of filter
            binning: Used binning in X & Y
            count: Number of flats to take.
            start_time: Time in seconds to start after the time set in init()
            readout: Time in seconds for readout per flat

        Returns:
            Estimated duration in seconds
        """

        # loop through images and add estimated exposure times at their respective start times
        elapsed = start_time
        for i in range(count):
            elapsed += self.exp_time(filter_name, binning, elapsed) + readout

        # we started at start_time, so subtract it again
        return elapsed - start_time


class SchedulerItem:
    """A single item in the flat scheduler"""

    def __init__(self, start: float, end: float, filter_name: str, binning: int, priority: float):
        """Initializes a new scheduler item

        Args:
            start: Start time in seconds
            end: End time in seconds
            filter_name: Name of filter
            binning: Used binning in X and Y
        """
        self.start = start
        self.end = end
        self.filter_name = filter_name
        self.binning = binning
        self.priority = priority

    def __repr__(self):
        """Nice string representation for item"""
        return '%d - %d (%s %dx%d): %.2f' % (self.start, self.end, self.filter_name,
                                             self.binning, self.binning, self.priority)


class Scheduler:
    """Scheduler for taking flat fields"""
    __module__ = 'pyobs.utils.skyflats'

    def __init__(self, functions: dict, priorities: SkyflatPriorities, observer: Observer, min_exptime: float = 0.5,
                 max_exptime: float = 5, timespan: float = 7200, filter_change: float = 30, count: int = 20,
                 combine_binnings: bool = True, readout: dict = None):
        """Initializes a new scheduler for taking flat fields

        Args:
            functions: Flat field functions
            priorities: Class handling priorities
            observer: Observer to use
            min_exptime: Minimum exposure time for flats
            max_exptime: Maximum exposure time for flats
            timespan: Timespan from now that should be scheduled [s]
            filter_change: Time required for filter change [s]
            count: Number of flats to schedule
            combine_binnings: Whether different binnings use the same functions.
            readout: Dictionary with readout times (in sec) per binning (as BxB).
        """
        self._eval = ExpTimeEval(observer, functions, combine_binnings=combine_binnings)
        self._observer = observer
        self._priorities = priorities
        self._min_exptime = min_exptime
        self._max_exptime = max_exptime
        self._schedules = []
        self._current = 0
        self._timespan = timespan
        self._filter_change = filter_change
        self._count = count
        self._readout = {} if readout is None else readout

    def __call__(self, time: Time):
        """Calculate schedule starting at given time

        Args:
            time: Time to start schedule at
        """

        # init evaluator
        self._eval.init(time)

        # sort filters by priority
        priorities = sorted(self._priorities().items(), key=operator.itemgetter(1), reverse=True)

        # place them
        schedules = []
        for task, priority in priorities:
            # find possible time
            self._find_slot(schedules, *task, priority)

        # sort by start time
        self._schedules = sorted(schedules, key=lambda x: x.start)

    def __iter__(self):
        """Iterator for scheduler items"""
        self._current = 0
        return self

    def __next__(self) -> SchedulerItem:
        """Iterate over scheduler items"""
        if self._current < len(self._schedules):
            item = self._schedules[self._current]
            self._current += 1
            return item
        else:
            raise StopIteration

    def _find_slot(self, schedules: list, filter_name: str, binning: int, priority: float):
        """Find a possible slot for a given filter/binning in the given schedule

        Args:
            schedules: List of existing schedules
            filter_name: Name of filter
            binning: Used binning
        """

        # get readout time
        sbin = '%dx%d' % (binning, binning)
        readout = self._readout[sbin] if sbin in self._readout else 0.

        # find first possible start time
        time = 0
        while time < self._timespan:
            # get exposure time
            exp_time_start = self._eval.exp_time(filter_name, binning, time)

            # are we in allowed limit?
            if self._min_exptime <= exp_time_start <= self._max_exptime:
                # seems to fit, get duration
                duration = self._eval.duration(filter_name, binning, self._count, start_time=time, readout=readout)

                # add time for filter change
                duration += self._filter_change

                # get exp time at end
                exp_time_end = self._eval.exp_time(filter_name, binning, time + duration)

                # still in limits?
                if self._min_exptime <= exp_time_end <= self._max_exptime:
                    # check for overlap with existing schedule
                    if not self._overlaps(schedules, time, time + duration):
                        # add schedule and quit
                        schedules.append(SchedulerItem(time, time + duration, filter_name, binning, priority))
                        return

            # next step
            time += 10

        # being here means we didn't find any
        return

    def _overlaps(self, schedules: list, start: float, end: float) -> bool:
        """Checks, whether a new scheduler item would overlap an existing item

        Args:
            schedules: List of existing scheduler items
            start: Start time of new item
            end: End time of new item

        Returns:
            Whether it overlaps
        """

        # loop all scheduler items
        item: SchedulerItem
        for item in schedules:
            # does it overlap?
            if (start < item.end and end > item.start) or (item.start < end and item.end > start):
                return True

        # no overlap found
        return False


__all__ = ['SchedulerItem', 'Scheduler']
