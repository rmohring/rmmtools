import pandas as pd


def convert_cols(
    df: pd.DataFrame,
    strcols=None,
    intcols=None,
    floatcols=None,
    boolcols=None,
    datecols=None,
    skipcols=None,
    date_format: str = "%Y-%m-%dT%H:%M:%S",
    date_utc=False,
) -> pd.DataFrame:
    """Converts specified columns in a DataFrame to appropriate data types.

    This function returns a modified version of a given Pandas DataFrame after
    converting specified columns to string, integer, float, boolean, or datetime
    formats. Columns that are not specified are assumed to be in floatcols

    Args:
        df (pd.DataFrame): The input DataFrame to be worked on
        strcols (list, optional): List of columns to be converted to strings
        intcols (list, optional): List of columns to be converted to integers
        floatcols (list, optional): List of columns to be converted to floats. If
            None, all columns not specified in the args (as intcols, skipcols, etc)
            are assumed to be floatcols.
        boolcols (list, optional): List of columns to be converted to boolean values
        datecols (list, optional): List of columns to be converted to datetime format
        skipcols (list, optional): List of columns to be excluded from automatic
            float detection in the case where floatcols=None.
        date_format (str, optional): Date format string for parsing datetime columns.
            Defaults to "%Y-%m-%dT%H:%M:%S"
        date_utc (bool, optional): Whether to convert datetime columns to UTC.
            Defaults to False.

    Returns:
        pd.DataFrame: The modified DataFrame with converted column types

    Notes:
        - Columns in `floatcols` and `intcols` are coerced into numeric values,
            replacing invalid entries with NaN
        - Columns in `datecols` are converted to datetime using the specified
            `date_format` and `date_utc` settings
        - Columns in `boolcols` are converted to False if in the list of values
             (`"N"`, `"False"`, `False`, `"false"`, `"0"`, `0`)
        - If `floatcols` is not provided, it is inferred by excluding
            specified `intcols`, `datecols`, `strcols`, `boolcols`, and `skipcols`
    """
    if strcols is None:
        strcols = []
    if datecols is None:
        datecols = []
    if intcols is None:
        intcols = []
    if boolcols is None:
        boolcols = []
    if skipcols is None:
        skipcols = []
    if floatcols is None:
        floatcols = [
            x
            for x in df.columns
            if (x not in (intcols + datecols + strcols + boolcols + skipcols))
        ]
    for c in floatcols + intcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in datecols:
        df[c] = pd.to_datetime(df[c], format=date_format, utc=date_utc, errors="raise")
    for c in boolcols:
        df[c] = df[c].apply(
            lambda x: False if x in ["N", "False", False, "false", "0", 0] else True
        )
    return df


def dataframe_report(df, num=7, c_thresh=30):
    """
    num:       How many itmes to show in value_counts for each column
    c_thresh:  Number of unique values equal to or below which the
               report will list a given column as 'categorizable'.

    Prints the top _num_ value_counts, number of uniques, and number
    of nan/null values for each column. Ends with a summary of columns
    with only one unique value, those that are all nan, and a list that
    might be candidates to be converted into the categorical dtype.
    """
    uniques = {}
    nans = {}
    totlen = len(df)

    for x in df.columns:
        series = df[x]
        print(f"\n---{x}---")
        print(series.value_counts().head(num))
        print(f" Column dtype: {series.dtype}")
        nansum = sum(series.isna())
        print(f" {nansum} NaNs/null found")
        nans[x] = nansum
        try:
            uniq = len(series.unique())
            print(f" {uniq} uniques")
            uniques[x] = uniq
        except TypeError as e:
            print(f"Caught {e}")
            uniques[x] = 0
            continue

    print("===============")
    print("Columns that only have one value:")
    print(f" {[x for (x,n) in uniques.items() if n==1]}")

    print("\nColumns that are all nan/null:")
    print(f" {[x for (x,n) in nans.items() if n==totlen]}")

    print(f"\nCategorical variable candidates (thresh={c_thresh}):")
    print(f" {[ x for (x,n) in uniques.items() if n<=c_thresh ]}")

    return


def all_indexes_except(full_index, except_index):
    """
    Return a list of all indexes except the specified item or items.  If
    except_index is not present, it returns the full list.
    """
    if isinstance(except_index, list):
        return_val = [x for x in full_index.names if x not in except_index]
    else:
        return_val = [x for x in full_index.names if x != except_index]

    return return_val
