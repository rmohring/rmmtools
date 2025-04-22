#!/usr/bin/env python
# Author: Rick Mohring (rmmohring@live.com)

import pandas as pd
import numpy as np
from functools import partial
from typing import Union, Iterable, List, Any
from ..util.general import check_not_none, listify, get_val_or_alt_or_raise


class AccessorBase(object):
    def __init__(self, data):
        self._data = data
        self._LOOKUP_DICT = {}

    def _verify_one_thing_only(self, field: str = None) -> None:
        """Raises a ValueError if there is more than one unique value in
        the specified field.
        """
        if self._data[field].nunique() != 1:
            clist = list(self._data[field].unique())[:3]
            raise ValueError(
                f"Can only handle one thing a time, first few are: {clist}"
            )
        return

    # Convenience methods to access subsets of dataframe
    def _make_picker(self, attr_name: str, field: str = None, **kwargs) -> None:
        """Factory method for creating custom picker methods.

        Args:
            attr_name (str): the new method name
            field (str): The DataFrame field to use as default for filtering
                in the generated picker method.
        """
        check_not_none([attr_name, field])

        setattr(
            self, attr_name, partial(self.pick, field=field, return_df=True, **kwargs)
        )

        # Dynamically set the docstring
        docstring = f"""
        The '{attr_name}' method is a picker that returns the subset of the DataFrame
        where the value(s) in field '{field}' are filtered by the specified `val`
        (which can be a single value or a list-like). This is a factory-generated
        method.

        Args:
            val (Any, or list-like of Any): the item or items that is
                looked for inside the column.
            field (optional, default {field}): The field to use for filtering,
                which can be overridden here.
            make_strings (bool, default False): forcibly cast the val or
                list of val to be strings
            invert (bool, default False): If True, picker will return the
                records where none of the `val` values are found in column
                specified by `field`.

        Returns:
            A filtered pandas DataFrame.
        """
        setattr(getattr(self, attr_name), "__doc__", docstring)

    # def _generic_pick_method(
    #     self, val, field=None, make_strings=False, invert=False
    # ) -> pd.Series:
    #     check_not_none(val)
    #     return self.pick(val, field=field, make_strings=make_strings, invert=invert)

    def true_pick(self) -> pd.Series:
        """Generate a series of all "True" values with the same index as the
        current series/dataframe (and obviously of the same length, as well).

        Returns:
            pd.Series: All bool True
        """
        return pd.Series(data=True, index=self._data.index)

    # Dataframe single query
    def pick(
        self,
        val: Any,
        field: str = None,
        make_strings: bool = False,
        invert: bool = False,
        above: int = 0,
        below: int = 0,
        return_df=False,
    ) -> pd.Series:
        """Generic picker, returns subset of dataframe with field values in my_list"""
        check_not_none([val, field])
        the_pick = self._data[field].isin(listify(val, make_strings=make_strings))

        return self.generic_pick(
            pick=the_pick, above=above, below=below, invert=invert, return_df=return_df
        )

    def generic_pick(
        self,
        pick: pd.Series = None,
        above: int = 0,
        below: int = 0,
        invert: bool = False,
        return_df=False,
    ) -> pd.Series:
        """Generic picker, returns subset of dataframe allowing invert, above,
        and below.
        """
        if pick is None:
            pick = self.true_pick()

        the_pick = pick.copy()

        if invert:
            the_pick = ~the_pick

        if above > 0:
            for i in range(1, above + 1):
                the_pick |= the_pick.shift(-i)

        if below > 0:
            for i in range(1, below + 1):
                the_pick |= the_pick.shift(i)

        if not return_df:
            retval = the_pick
        else:
            retval = self._data.loc[the_pick, :]
        return retval

    def mpick(self, invert=False, make_strings=False, as_or=False, **kwargs):
        the_pick = self.true_pick()
        for field, val in kwargs.items():
            if not as_or:
                the_pick &= self._data[field].isin(
                    listify(val, make_strings=make_strings)
                )
            else:
                the_pick |= self._data[field].isin(
                    listify(val, make_strings=make_strings)
                )

        if invert:
            the_pick = ~the_pick

        return the_pick

    def get(self, invert=False, make_strings=False, as_or=False, **kwargs):
        the_pick = self.mpick(
            invert=invert, make_strings=make_strings, as_or=as_or, **kwargs
        )

        return self._data.loc[the_pick, :]

    # Field translations
    def translate_fields(self, field=None, lookup=None):
        if field is None:
            fields = self._data.columns
        else:
            fields = listify(field)

        lookup = get_val_or_alt_or_raise(lookup, self._LOOKUP_DICT)

        retval = [(i, f, lookup.get(f, "UNKNOWN")) for (i, f) in enumerate(fields)]
        if len(retval) == 1:
            retval = retval[0]

        return retval

    # Simple comparative functions, returns a subset of the series meeting the condition
    def gt(self, val=None):
        '''Returns the sub-item where its values are greater than "val"'''
        check_not_none(val)
        return self._data[self._data > val]

    def lt(self, val=None):
        '''Returns the sub-item where its values are less than "val"'''
        check_not_none(val)
        return self._data[self._data < val]

    def ge(self, val=None):
        '''Returns the sub-item where its values are greater than or equal to "val"'''
        check_not_none(val)
        return self._data[self._data >= val]

    def le(self, val=None):
        '''Returns the sub-item where its values are less than or equal to "val"'''
        check_not_none(val)
        return self._data[self._data <= val]

    def eq(self, val=None):
        '''Returns the sub-item where its values are equal to "val"'''
        check_not_none(val)
        return self._data[self._data == val]

    def ne(self, val=None):
        '''Returns the sub-item where its values are not equal to "val"'''
        check_not_none(val)
        return self._data[self._data != val]

    def nonzero(self):
        """Returns the sub-item where its values are not equal to zero"""
        return self.ne(0)

    def mid(self, num=5, start=0.5):
        """Similar to head() or tail(), but gets records from the middle.
        If start is integer it begins at that record.
        If start is non-integer, it treats it as a fraction of the
           full length and begins there, defaults to the middle record.
        """
        if not (isinstance(start, (int, np.integer))):
            start = int(start * len(self._data))

        return self._data.head(start + num).tail(num)

    def head(self, *args, **kwargs):
        """Simple pass-through to df.head() for symmetry"""
        return self._data.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Simple pass-through to df.tail() for symmetry"""
        return self._data.tail(*args, **kwargs)

    # Properties
    @property
    def length(self):
        """Length of item"""
        return len(self._data)

    @property
    def nancount(self) -> int:
        """Count of isna() values"""
        return (self._data.isna()).sum()

    @property
    def nonnancount(self) -> int:
        """Count of ~isna() values"""
        return (~self._data.isna()).sum()

    @property
    def zerocount(self) -> int:
        """Count of zero values"""
        return (self._data == 0).sum()

    # Transforms
    def log10(self):
        return np.log10(self._data)

    def log1p(self):
        return np.log1p(self._data)

    def expm1(self):
        return np.expm1(self._data)

    def trim_data(self, **kwargs):
        data = self._data.copy()

        if kwargs.get("dropna", False):
            data = data.dropna()

        start = kwargs.get("start", 0)
        end = kwargs.get("end", len(data))

        return data.iloc[start:end].copy()
