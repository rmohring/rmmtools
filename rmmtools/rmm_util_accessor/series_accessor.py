import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union

from .accessor_base import AccessorBase
from ..util.general import check_not_none

from pandas.api.extensions import register_series_accessor


@register_series_accessor("rmm")
class RMM_SeriesUtil(AccessorBase):
    def __init__(self, data):
        super().__init__(data)

    @property
    def intspan(self) -> int:
        """An integer version of the max - min values"""
        return int(self._data.max() - self._data.min())

    # Isna
    def isna(self) -> pd.Series:
        """Returns the sub-Series where its values are isna()"""
        return self._data[self._data.isna()]

    def notisna(self) -> pd.Series:
        """Returns the sub-Series where its values are not isna()"""
        return self._data[~self._data.isna()]

    # String functions
    def str_contains(self, thing: str, regex: bool = False, **kwargs) -> pd.Series:
        """Returns the sub-Series where its values contain the thing.

        Passes regex parameter through, but defaults to False
        because it's faster that way for normal strings.

        Args:
            thing (str, list-like of strings): the item or items that is
                looked for inside the column.
            regex (bool, optional): Pass-through to the .str accessor
                that determines whether or not the `thing` is interpreted
                as a regular expression.

        Returns:
            A pandas Series.
        """
        data = self._data.dropna()

        if isinstance(thing, pd.Series) or isinstance(thing, np.ndarray):
            thing = thing.tolist()

        if isinstance(thing, list) or isinstance(thing, tuple):
            thing = "|".join(thing)
            regex = True

        return data[data.str.contains(thing, regex=regex, **kwargs)]

    def str_startswith(self, thing: str) -> pd.Series:
        """Returns the sub-series where its values start with `thing`.

        Args:
            thing (str): the item or items that is
                looked for inside the column.

        Returns:
            A pandas Series.
        """
        data = self._data.dropna()
        return data[data.str.startswith(thing)]

    def str_endswith(self, thing: str) -> pd.Series:
        """Returns the sub-series where its values end with the thing.

        Args:
            thing (str): the item or items that is
                looked for inside the column.

        Returns:
            A pandas Series.
        """
        data = self._data.dropna()
        return data[data.str.endswith(thing)]

    # Sorting
    def sort(self, ascending: bool = True, **kwargs) -> pd.Series:
        """Returns the series sorted by values. Same as `.sort_values()`.

        Args:
            ascending (bool): Direction of sorting, defaults to
                lowest-to-highest. Also see `.rsort()`.

        Returns:
            A pandas Series.
        """
        return self._data.sort_values(ascending=ascending, **kwargs)

    def rsort(self, **kwargs) -> pd.Series:
        """Returns the sorted values in descending order.
            See `.sort()`.

        Returns:
            A pandas Series.
        """
        return self.sort(ascending=False, **kwargs)

    # Transitions
    def get_transitions(self, offset: int = 1) -> pd.Series:
        """Returns a sub-series of values in the series that differ
        from the next value (with `offset==1`).

        Args:
            offset (int, optional): Defaults to 1, but can be changed to compare
                different offsets (e.g., for daily data, and `offset==7`
                would compare data by day of week.)

        Returns:
            A pandas Series.
        """
        return self._data.loc[self._data != self._data.shift(offset)]

    # Top, bottom, value counting
    def top(self, n: int = 5) -> List:
        """Returns a list of the top n values by value counts from the items
        in the series.
        """
        return list(
            self._data.value_counts().sort_values(ascending=False).head(n).index.values
        )

    def is_in_top(self, n: int = 5) -> pd.Series:
        """Returns boolean Series of whether each item is in the top n
        values by value counts
        """
        return self._data.isin(self.top(n=n))

    def in_top(self, n: int = 5) -> pd.Series:
        """Returns Series where each item is in the top n values by value counts"""
        return self._data[self.is_in_top(n=n)]

    def bottom(self, n: int = 5) -> List:
        """Returns a list of the bottom n values by value counts from the items
        in the series.
        """
        return list(
            self._data.value_counts().sort_values(ascending=False).tail(n).index.values
        )

    def is_in_bottom(self, n: int = 5) -> pd.Series:
        """Returns boolean Series of whether each item is in the bottom n
        values by value counts
        """
        return self._data.isin(self.bottom(n=n))

    def in_bottom(self, n: int = 5) -> pd.Series:
        """Returns Series where each item is in the bottom n values
        by value counts.
        """
        return self._data[self.is_in_bottom(n=n)]

    def vc(self, dropna: bool = False, **kwargs) -> pd.Series:
        """Returns value_counts() for the perpetually lazy...defaults
        to including nans
        """
        return self._data.value_counts(dropna=dropna, **kwargs)

    def vcvc(self, dropna: bool = False, **kwargs) -> pd.Series:
        """Returns value_counts().value_counts() for the perpetually
        lazy
        """
        return self._data.value_counts(dropna=dropna, **kwargs).value_counts(
            dropna=dropna, **kwargs
        )

    # Date-related
    def date_between(
        self,
        start: Optional[Union[str, np.datetime64]] = None,
        end: Optional[Union[str, np.datetime64]] = None,
        eopen: bool = True,
    ) -> pd.Series:
        """[summary]

        Args:
            start (datelike, optional): Start date. Defaults to None.
            end (datelike, optional): End date. Defaults to None.
            eopen (bool, optional): If True, does not include the end date.
                Defaults to True.

        Raises:
            ValueError: [description]

        Returns:
            pd.Series: [description]
        """
        check_not_none(
            [start, end], msg="Need to specify at least one of the start or end date"
        )

        cut = self.true_pick()

        if start is not None:
            cut &= self._data >= (pd.to_datetime(start))

        if end is not None:
            if not eopen:
                cut &= self._data < (pd.to_datetime(end) + pd.Timedelta(1, "d"))
            else:
                cut &= self._data < (pd.to_datetime(end))
        return self._data[cut]

    def to_datetime(
        self, format: str = "%Y-%m-%d", errors: str = "coerce", **kwargs
    ) -> pd.Series:
        """Converts a series into a pandas datetime object"""
        return pd.to_datetime(self._data, format=format, errors=errors, **kwargs)

    def missing_dates(
        self,
        start: Optional[Union[str, np.datetime64]] = None,
        end: Optional[Union[str, np.datetime64]] = None,
    ) -> pd.DatetimeIndex:
        """Returns a DatetimeIndex that shows which dates are missing from the
        given Series vs. the sequential list of dates from start to end. Start
        and end dates can be strings or other datelikes.

        Args:
            start (datelike, optional): Start date.
            end (datelike, optional): End date.

        Returns:
            pd.DatetimeIndex: [description]
        """
        if start is None:
            start = self._data.min()

        if end is None:
            end = self._data.max()

        date_index = pd.date_range(start=start, end=end, freq="D")
        return date_index[~date_index.isin(self._data)]

    # Transforms
    def normalize(self, scale: float = 1) -> pd.Series:
        """Returns the series normalized by its sum

        Args:
            scale (float, optional): Optionally multiply the result by a
            scale factor (default = 1)
        """
        return self._data / self._data.sum() * scale

    def blur_uniform(self, low: float = -0.5, high: float = 0.5) -> pd.Series:
        """Returns the series "blurred" by a uniform random number.

        Useful for making scatter plots, etc, where having some jitter can help
        vizualization. For integer values, defaults to using -0.5, 0.5 as the
        low, high values for the uniform distribution.

        Args:
            low (float, optional): Low end of the blur.  See numpy.random.uniform().
            high (float, optional): High end of the blur

        See also:
            rmm.blur(), rmm.blur_normal()
        """
        return self.blur(dist="uniform", low=low, high=high)

    def blur_normal(self, loc: float = 0.0, scale: float = 1.0) -> pd.Series:
        """Returns the series "blurred" by a normally-distributed random number.

        Useful for making scatter plots, etc, where having some jitter can help
        vizualization. For integer values, defaults to using 0, 1 as the mean
        location and scale for the normal distribution.

        Args:
            loc (float, optional): Mean value for the center of the
                normal distribution. See numpy.random.normal().
            scale (float, optional): Width of the normal distribution.

        See also:
            rmm.blur(), rmm.blur_uniform()
        """
        return self.blur(dist="normal", loc=loc, scale=scale)

    def blur(self, dist: Optional[str] = None, **kwargs) -> pd.Series:
        """Returns the series "blurred" by a random number.  The methods
        and parameters or np.random are available.  Defaults to a uniform
        distribution with low/high (-0.5, 0.5).

        Useful for making scatter plots, etc, where having some jitter can help
        vizualization. For integer values, defaults to using 0, 1 as the mean
        location and scale for the normal distribution.

        See also:
            rmm.blur_uniform(), rmm.blur_normal()
        """
        if dist is None:
            dist = "uniform"
            kwargs["low"] = kwargs.get("low", -0.5)
            kwargs["high"] = kwargs.get("high", 0.5)

        return self._data.apply(lambda x: x + getattr(np.random, dist)(**kwargs))

    # Visualization and helpers
    def get_hist_bins(
        self,
        bins: int = 100,
        range: Tuple[float] = None,
        intrange: Tuple[int] = None,
        density: bool = False,
        log: bool = False,
        log1p: bool = False,
        dropna: bool = False,
        **kwargs,
    ) -> Tuple:
        """Returns the bins and counts for a histogram of the series.

        Args:
            bins (int, optional): Number of histogram bins (see np.histogram)
            range (tuple, optional): Range for histogram (see np.histogram)
            intrange: tuple, Given as (start_bin, end_bin), overrides bins
                    and range options to give one bin per unit of range.
            density: Boolean, whether or not to make the integral over the range
                    equal 1 (see np.histogram)

            log:   Boolean, whether or not to use log-10 scale (i.e., apply
                np.log10 to data)
            log1p: Boolean, whether or not to apply np.log1p to data.
                Takes precedence over log.
            dropna: Boolean, whether or not to drop NaN values from the series

            **kwargs: Other keyword args that will be ignored
        """

        data = self._data.copy()
        data.name = str(data.name)

        # if intbins is not None:
        #    bins = intbins[1]
        #    range = (intbins[0]-0.5, intbins[0]+intbins[1]-0.5)

        if intrange is not None:
            bins = intrange[1] - intrange[0] + 1
            range = (intrange[0] - 0.5, intrange[1] + 0.5)

        if dropna:
            ff, ee = np.histogram(
                data.dropna(), bins=bins, range=range, density=density
            )
        else:
            ff, ee = np.histogram(data, bins=bins, range=range, density=density)

        if log1p:
            ff = np.log1p(ff)
        elif log:
            ff = np.log10(ff)

        return (ff, ee)

    # Pretty printing
    def tabu(
        self, num: int = None, clip: int = 50, quiet: bool = False, **kwargs
    ) -> None:
        """Displays the dataframe using the tabulate module for
        pretty printing. Specify how many rows to show. Requires the
        tabulate module to be installed.

        Args:
            num (int, optional): Number of rows to print, similar to .head().
            clip (int, optional): If `num` isn't specified, it clips the
                length at `clip` rows to prevent long time delays in output.
                If set to None, it will not clip. Defaults to 50 rows.
            quiet (bool, optional): If True, will suppress warning to user
                that output is clipped at `clip` rows. Default is False.
            **kwargs pass through to tabulate()

        Returns:
            Prints output to screen, no return value.
        """
        from tabulate import tabulate

        opts_dict = dict(headers="keys", tablefmt="simple")
        opts_dict.update(kwargs)

        data = self._data.to_frame()

        if (num is None) and (clip is not None):
            length = len(self._data)
            if length > clip:
                n = clip
                if not quiet:
                    print(f"Note: Only showing {n} / {length} rows")
            else:
                n = length
        else:
            n = num

        print(tabulate(data[:n], **opts_dict))

        return
