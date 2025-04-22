import pandas as pd
import numpy as np
from typing import Iterable, List, Optional, Union

from .accessor_base import AccessorBase
from ..util.general import check_not_none, listify

from pandas.api.extensions import register_dataframe_accessor

OneOrMoreCols = Union[str, Iterable[str]]


@register_dataframe_accessor("rmm")
class RMM_DataFrameUtil(AccessorBase):
    def __init__(self, data):
        super().__init__(data)

    # Properties
    @property
    def column_list(self) -> List:
        return self._data.columns.tolist()

    # Isna
    def isna(self, cols: OneOrMoreCols = None, all: bool = True) -> pd.DataFrame:
        """Returns a subset DataFrame with only rows that have
        at least one NaN value. If `all=True`, it requires all values to
        be NaN. If `all=False`, it requires at least one NaN in a row.

        Args:
            cols (OneOrMoreCols, optional): Column str or list.
                Defaults to all columns.
            all (bool, optional): Toggles np.all vs. np.any
        """
        if cols is None:
            cols = self.column_list
        if all:
            return self._data.loc[np.all(self._data[listify(cols)].isna(), axis=1), :]
        else:
            return self._data.loc[np.any(self._data[listify(cols)].isna(), axis=1), :]

    def notisna(self, cols: OneOrMoreCols = None, all: bool = True) -> pd.DataFrame:
        """Returns a subset DataFrame with only rows that have
        zero NaN values. If `all=True`, it requires all values to
        be non-NaN. If `all=False`, it requires at least one non-NaN in a row.

        Args:
            cols (OneOrMoreCols, optional): Column str or list.
                Defaults to all columns.
            all (bool, optional): Toggles np.all vs. np.any
        """

        if cols is None:
            cols = self.column_list

        if all:
            return self._data.loc[np.all(~self._data[listify(cols)].isna(), axis=1), :]
        else:
            return self._data.loc[np.any(~self._data[listify(cols)].isna(), axis=1), :]

    # String functions
    def str_contains(
        self,
        field: str = None,
        thing: Union[str, int, float] = None,
        regex: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Returns the sub-dataframe where the field values contain the thing.

        Passes regex parameter through, but here defaults to False
        because it's faster that way for normal strings
        """
        check_not_none(
            [field, thing],
            "On a dataframe, must specify both the field and what to seek",
        )

        idx = (
            self._data[field].rmm.str_contains(thing=thing, regex=regex, **kwargs).index
        )
        return self._data.loc[idx, :]

    def str_startswith(self, field: str = None, thing: str = None) -> pd.DataFrame:
        """Returns the sub-dataframe where the field values start with the thing."""
        check_not_none(
            [field, thing],
            "On a dataframe, must specify both the field and what to seek",
        )

        idx = self._data[field].rmm.str_startswith(thing=thing).index
        return self._data.loc[idx, :]

    def str_endswith(self, field: str = None, thing: str = None) -> pd.DataFrame:
        """Returns the sub-dataframe where the field values end with the thing."""
        check_not_none(
            [field, thing],
            "On a dataframe, must specify both the field and what to seek",
        )

        idx = self._data[field].rmm.str_endswith(thing=thing).index
        return self._data.loc[idx, :]

    # Sorting
    def sort(
        self, by: Union[str, Iterable] = None, ascending: bool = True
    ) -> pd.DataFrame:
        """Returns the sorted values by the specified field(s)"""
        return self._data.sort_values(by=by, ascending=ascending)

    def rsort(self, by: Union[str, Iterable] = None):
        """Returns the sorted values by the specified field(s),
        but defaults to descending order.
        """
        return self.sort(by=by, ascending=False)

    # Transitions
    def get_transitions(
        self, key_cols: OneOrMoreCols, full: bool = False
    ) -> pd.DataFrame:
        """Returns the sub-dataframe showing values that differ from the next value"""
        c = listify(key_cols)
        if full:
            rval = self._data.loc[
                np.any(self._data[c] != self._data[c].shift(1), axis=1), :
            ]
        else:
            rval = self._data.loc[
                np.any(self._data[c] != self._data[c].shift(1), axis=1), c
            ]
        return rval

    # Top, bottom, value counting
    def top(self, col: str = None, n: int = 5) -> List:
        """Returns a list of the top n values by value counts from the items
        in the specified column.
        """
        check_not_none(col)
        return self._data[col].rmm.top(n=n)

    def is_in_top(self, col=None, n=5) -> pd.Series:
        """Returns boolean Series of whether each item is in the top n
        values by value counts in the specified column.
        """
        check_not_none(col)
        return self._data[col].rmm.is_in_top(n=n)

    def in_top(self, col=None, n=5) -> pd.DataFrame:
        """Returns Dataframe containing rows where each item is in the
        top n values by value counts for the specified column.
        """
        check_not_none(col)
        return self._data[self._data[col].rmm.is_in_top(n=n)]

    def bottom(self, col: str = None, n: int = 5) -> List:
        """Returns a list of the bottom n values by value counts from the items
        in the specified column.
        """
        check_not_none(col)
        return self._data[col].rmm.bottom(n=n)

    def is_in_bottom(self, col: str = None, n: int = 5) -> pd.Series:
        """Returns boolean Series of whether each item is in the bottom n
        values by value counts in the specified column.
        """
        check_not_none(col)
        return self._data[col].rmm.is_in_bottom(n=n)

    def in_bottom(self, col: str = None, n: int = 5) -> pd.Series:
        """Returns Dataframe containing rows where each item is in the
        bottom n values by value counts for the specified column.
        """
        check_not_none(col)
        return self._data[self._data[col].rmm.is_in_bottom(n=n)]

    # Date-related
    def date_between(
        self,
        field: str = None,
        start: Optional[Union[str, np.datetime64]] = None,
        end: Optional[Union[str, np.datetime64]] = None,
        eopen: bool = False,
    ) -> pd.DataFrame:
        if field is None:
            raise ValueError("Need to specify field to use as the date")

        idx = (
            self._data[field].rmm.date_between(start=start, end=end, eopen=eopen).index
        )
        return self._data.loc[idx, :]

    def missing_dates(
        self,
        field: str = None,
        start: Optional[Union[str, np.datetime64]] = None,
        end: Optional[Union[str, np.datetime64]] = None,
    ) -> pd.DatetimeIndex:
        """Returns a DatetimeIndex that shows which dates are missing from the
        specified column vs. the sequential list of dates from start to end. Start
        and end dates can be strings or other datelikes.

        Args:
            start (datelike, optional): Start date.
            end (datelike, optional): End date.

        Returns:
            pd.DatetimeIndex: [description]
        """

        check_not_none(field, msg="Need to specify field to use as the date")
        return self._data[field].rmm.missing_dates(start=start, end=end)

    def date_counts(
        self, field: str = None, resamp: str = "D", normalize: bool = False
    ) -> pd.Series:
        if not normalize:
            ret_val = self._data.groupby(field).size().rename("cts")
        else:
            ret_val = (
                self._data.groupby(self._data[field].dt.normalize())
                .size()
                .rename("cts")
            )

        if resamp is not None:
            ret_val = ret_val.resample(resamp).sum()

        return ret_val

    # Columns: listing, reassignment, manipulation
    def columns_except(self, cols: Union[str, Iterable[str]]) -> List:
        tmp_cols = listify(cols)
        badcols = [c for c in tmp_cols if c not in self._data.columns]
        if badcols:
            raise ValueError(f"{badcols} not in df columns")
        return [c for c in self._data.columns if c not in tmp_cols]

    def columns_containing(
        self, fragment: Union[str, list] = None, invert: bool = False
    ) -> List:
        if fragment is None:
            raise ValueError(f"Need to specify fragment(s) to look for")

        frag = listify(fragment)
        result = []
        for c in self._data.columns:
            for f in frag:
                if f in c:
                    result.append(c)
                    break

        if invert:
            result = self.columns_except(result)

        return result

    def columns_plus_suffix(
        self, cols: OneOrMoreCols = None, suffix: str = "_copy"
    ) -> List:
        """Return a list of the specified columns with a suffix concatenated.
        Useful for renaming/copying a subset of columns, or adding the result
        of a groupby.transform() to the initial dataframe.

        Args:
            cols (Union[str, Iterable[str]], optional): One or more columns.
                Defaults to None.
            suffix (str, optional): Suffix to append. Defaults to "_copy".
        """
        if cols is None:
            cols = self._data.columns
        tmp_cols = listify(cols)
        newcols = [f"{c}{suffix}" for c in tmp_cols]

        return newcols

    def allbut(self, cols: OneOrMoreCols) -> pd.DataFrame:
        """Return a dataframe with all but the specified columns"""
        return self._data[self.columns_except(cols)]

    def rearrange_cols(
        self,
        first: Optional[Union[str, Iterable[str]]] = None,
        last: Optional[Union[str, Iterable[str]]] = None,
        drop: Optional[Union[str, Iterable[str]]] = None,
    ) -> pd.DataFrame:
        tmp_first = listify(first)
        tmp_last = listify(last)
        tmp_drop = listify(drop)

        other_cols = [
            c for c in self._data.columns if c not in tmp_first + tmp_last + tmp_drop
        ]

        return self._data[tmp_first + other_cols + tmp_last]

    def concat_cols(
        self, col1: str = None, col2: str = None, sep: str = "_"
    ) -> pd.Series:
        """Takes the values of two columns and concatenates them into a
        resulting string Series joined by the `sep` separator. Also see
        `concat_mcols`.

        Args:
            col1 (str): Column 1 name.
            col2 (str): Column 2 name.
            sep (str): Defaults to "_".
        """
        check_not_none([col1, col2])
        return self.concat_mcols([col1, col2], sep=sep)

    def concat_mcols(self, cols: List[str], sep="_") -> pd.Series:
        """Takes the values of two columns and concatenates them into a
        resulting string Series joined by the `sep` separator. Also see
        `concat_mcols`.

        Args:
            cols (List[str]): List of column names.
            sep (str): Defaults to "_".
        """
        check_not_none(cols)
        if len(cols) < 2:
            raise ValueError("Must have list of at least two columns")

        val = self._data[cols[0]].astype(str)
        for col in cols[1:]:
            val += sep + self._data[col].astype(str)
        return val

    # Transforms, statistics, smoothing
    def _prep_for_transform(self, x=None, y=None, by=None, sort_by=None, data_in=None):
        data = self._data if data_in is None else data_in

        if sort_by is not None:
            data = data.sort_values(sort_by)

        if y is None:
            y = [c for c in data.columns if c not in listify(by) and c != x]

        keep_columns = listify(y) + listify(by)

        # x is None means that the assumption is that 'x' is already in the index
        # or, if it's supplied and it matches the index name, then it's already
        # in the index
        if (x is None) or (data.index.name == x):
            data = data[keep_columns]
            if by is not None:
                data = data.reset_index()

        # or if it's an unnamed index, then all the columns have to be present
        elif data.index.name is None:
            if by is None:
                data = data.set_index(x)[keep_columns]
            else:
                data = data[[x] + keep_columns]

        # possibly there's a MultiIndex... so if it gets here, just reset the
        # index and start from there
        elif by is None:
            data = data.reset_index()[[x] + keep_columns]
        else:
            data = data.reset_index().set_index(x)[keep_columns]

        if by:
            data = data.pivot_table(index=x, columns=listify(by), values=listify(y))

        return data.copy()

    def shift(
        self, x=None, y=None, lag=1, by=None, sort_by=None, to_frame=True, **kwargs
    ):
        """Returns a diff on the supplied y columns, indexed by x.
        Useful for chaining into an operation such as curve().
        """
        data = self._prep_for_transform(x, y, by=by, sort_by=sort_by)
        x = data.index.name

        rval = data.shift(lag)

        if by:
            rval = rval.stack(by)

        excludes = [x] + listify(by)

        newcols = []
        for c in rval.columns:
            if c not in excludes:
                if lag > 0:
                    newcols.append(f"{c}_lag_{lag}")
                elif lag <= 0:
                    newcols.append(f"{c}_fwd_{abs(lag)}")
            else:
                newcols.append(c)

        rval.columns = newcols

        if by:
            rval = rval.reset_index(by)

        if isinstance(rval, pd.Series) and to_frame:
            rval = rval.to_frame()
        return rval

    def diff(
        self,
        x=None,
        y=None,
        lag=1,
        by=None,
        sort_by=None,
        label="diff",
        to_frame=True,
        **kwargs,
    ):
        """Returns a diff on the supplied y columns, indexed by x.
        Useful for chaining into an operation such as curve().
        """
        data = self._prep_for_transform(x, y, by=by, sort_by=sort_by)
        x = data.index.name

        rval = data.diff(lag)

        if by:
            rval = rval.stack(by)

        excludes = [x] + listify(by)

        newcols = []
        for c in rval.columns:
            if c not in excludes:
                if lag == 1:
                    newcols.append(f"{c}_{label}")
                elif lag > 1:
                    newcols.append(f"{c}_{label}_lag_{lag}")
                elif lag <= 0:
                    newcols.append(f"{c}_{label}_fwd_{abs(lag)}")
            else:
                newcols.append(c)

        rval.columns = newcols

        if by:
            rval = rval.reset_index(by)

        if isinstance(rval, pd.Series) and to_frame:
            rval = rval.to_frame()
        return rval

    def rolling(
        self, x=None, y=None, window=None, by=None, agg="mean", to_frame=True, **kwargs
    ):
        """Returns a rolling window on the supplied y columns, indexed by x.
        Useful for chaining into an operation such as curve().
        """
        data = self._prep_for_transform(x, y, by=by)
        x = data.index.name

        rval = getattr(data.rolling(window=window, **kwargs), agg)()

        if by:
            rval = rval.stack(by)

        excludes = [x] + listify(by)

        newcols = []
        for c in rval.columns:
            if c not in excludes:
                newcols.append(f"{c}_rolling_{agg}_{window}")
            else:
                newcols.append(c)

        rval.columns = newcols

        if by:
            rval = rval.reset_index(by)

        if isinstance(rval, pd.Series) and to_frame:
            rval = rval.to_frame()
        return rval

    def ewm(
        self, x=None, y=None, span=None, by=None, agg="mean", to_frame=True, **kwargs
    ):
        """Returns an exponentially weighted smooth on the supplied y columns,
        indexed by x. Useful for chaining into an operation such as curve().
        """
        data = self._prep_for_transform(x, y, by=by)
        x = data.index.name

        opts_dict = kwargs.copy()

        if span is None:
            if "halflife" in kwargs:
                label = f"ewm_{agg}_halflife_{kwargs['halflife']}"
            elif "com" in kwargs:
                label = f"ewm_{agg}_com_{kwargs['com']}"
            elif "alpha" in kwargs:
                label = f"ewm_{agg}_alpha_{kwargs['alpha']}"
            else:
                raise ValueError("Must specify halflife, com, or alpha")
        else:
            label = f"ewm_{agg}_{span}"
            opts_dict.update({"span": span})

        opts_dict.update(kwargs)

        rval = getattr(data.ewm(**opts_dict), agg)()

        if by:
            rval = rval.stack(by)

        excludes = [x] + listify(by)

        newcols = []
        for c in rval.columns:
            if c not in excludes:
                newcols.append(f"{c}_{label}")
            else:
                newcols.append(c)

        rval.columns = newcols

        if by:
            rval = rval.reset_index(by)

        if isinstance(rval, pd.Series) and to_frame:
            rval = rval.to_frame()
        return rval

    # Pretty printing
    def tabu(
        self,
        num: int = None,
        clip: int = 50,
        cols: OneOrMoreCols = None,
        quiet: bool = False,
        **kwargs,
    ) -> None:
        """Displays the dataframe or a subset of `cols` columns using the
        tabulate module for pretty printing. Specify how many rows to show.
        Requires the tabulate module to be installed.

        Args:
            num (int, optional): Number of rows to print, similar to .head().
            clip (int, optional): If `num` isn't specified, it clips the
                length at `clip` rows to prevent long time delays in output.
                If set to None, it will not clip. Defaults to 50 rows.
            cols (str or list(str), optional): Limit which columns are
                output. Defaults to all columns.
            quiet (bool, optional): If True, will suppress warning to user
                that output is clipped at `clip` rows. Default is False.
            **kwargs pass through to tabulate()

        Returns:
            Prints output to screen, no return value.
        """
        from tabulate import tabulate

        opts_dict = dict(headers="keys", tablefmt="simple")
        opts_dict.update(kwargs)

        if cols is None:
            data = self._data
        else:
            data = self._data[listify(cols)]

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
