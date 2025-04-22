import numpy as np
import pandas as pd

#@profile
def mscore(x, groupby=None, window=16, center=True, cutoff=0, norm=0.6745):
    """
    Calculates the modified Z-score

    Boris Iglewicz and David Hoaglin (1993),
    "Volume 16: How to Detect and Handle Outliers",
    The ASQC Basic References in Quality Control: Statistical Techniques
    The normalization of 0.6745 is from that paper.
    """
    if groupby is not None:
        return x.groupby(groupby).apply(lambda g: mscore(g, window=window, 
                                        center=center, cutoff=cutoff, norm=norm))
    assert isinstance(x, pd.Series)

    median = x.rolling(window=window, center=center, min_periods=1).median()

    # median absolute deviation
    mad = (x - median).abs().median()

    if mad > 0.0:
        series = (x - median) * norm / mad
        if cutoff:
            series.loc[median < cutoff] = np.nan
    else:
        series = pd.Series(data = np.nan, index = x.index)

    series.name = 'mscore'

    return series

#@profile
def sn_score(x, groupby=None, window=7, center=False, cutoff=0, norm=1.1926):
    """
    Calculates the Sn-scale measure and the deviation from it
    in a rolling historical window.
    Alternatives to the Median Absolute Deviation
    Peter J. Rousseeuw and Christophe Croux
    Journal of the American Statistical Association
    December 1993, Vol. 88, No. 424, Theory and Methods
    The normalization of 1.1926 is from that paper.  
    """
    if groupby is not None:
        return x.groupby(groupby).apply(lambda g: sn_score(g, window=window,
                                        center=center, cutoff=cutoff, norm=norm))
    assert isinstance(x, pd.Series)

    def numerator_func(d):
        return pd.DataFrame(np.array(d)[:, None] - \
                            np.array(d)).abs().median().iloc[-1]
        
    def sn_func(d):
        return pd.DataFrame(np.array(d)[:, None] - \
                            np.array(d)).abs().median().median()

    def base_func(d):
        return pd.DataFrame(np.array(d)[:, None] - np.array(d)).abs().median()

    # Setting center=False forces this to always use causal data
    # in a training context where "future" historical data might be available
    # if the data is not pre-treated
    data = x.rolling(window=window, center=center, min_periods=1)

    numerator = data.apply(numerator_func, raw=True)
    sn = data.apply(sn_func, raw=True)
    #thing = data.apply(base_func, raw=True)
    #numerator = thing.iloc[-1]
    #sn = thing.median()

    rous = (numerator-sn)/sn
    
    # Experiment
    #last_diff = x - x.shift(1)
    #rous = (last_diff-sn)/sn

    #simple_median = data.median()
    #simple_mad = (x - simple_median).abs().median()
    #mscore = (x - simple_median) / simple_mad * 0.6745

    # This keeps it from evaluating a score in places where the counts are
    # very low (really that the scatter is low), so that it doesn't get 
    # suprious outliers.
    if cutoff:
        rous.loc[sn < cutoff] = np.nan

    rous.replace(np.inf, np.nan, inplace=True)
    rous.replace(-np.inf, np.nan, inplace=True)

    rous.fillna(0, inplace=True)

    series = rous * norm
    series.name = 'sn_score'

    return series

def get_daterange(x, start=None, end=None):
    assert isinstance(x, pd.Series) or isinstance(x, pd.DataFrame)
    x_new = x.copy()
    if start is not None:
        x_new = x_new.iloc[x_new.index.searchsorted(start):]
    if end is not None:
        x_new = x_new.iloc[:x_new.index.searchsorted(end)]
    return x_new

def interpolate(x, cut, groupby=None):
    if groupby is not None:
        return x.groupby(groupby).transform(lambda g: interpolate(g, cut))
    assert isinstance(x, pd.Series)
    series = x.copy()
    series.loc[cut] = np.nan
    series.interpolate(inplace=True, limit_direction='both')
    series = series.bfill()
    series.fillna(value=x, inplace=True)
    return series

def integerize(series_in, random_state=0):
    ''' Takes a series of float values and returns integer values that
        sum up to integer closest to the total. There is a stochastic
        step, so it will give different results depending on the random
        seed. This is typically used if an integer number of counts was 
        previously prorated into fractional bins. as opposed to rounding
        it avoids the issue of affecting the sum. For example, if there
        were 100 values of 0.1 each they would sum to 10, but would all 
        round to zero, so the "integer" sum would be zero. 
        
        The method is similar to a "chip race off" in poker. If each row
        has x+f counts where x is the integer portion, and 0 <= f < 1 is
        the fractional part, the resulting row gets a minimum of x 
        counts. The rounded sum of the f values for all rows determines 
        how many counts remain to distribute, and these are allocated
        stochastically to the rows with probabilities proportional to
        each row's f.  Because the sum is rounded, the sum of the 
        newly allocated counts will be within one count of the original     
        sum, but unless the original sum was an integer, it will not
        be exactly the same.
        
        Returns a new pd.Series.
    '''
    assert isinstance(series_in, pd.Series)

    df = series_in.copy().rename('orig').to_frame()
    df['floor'] = df['orig'].astype(int)
    
    df['rem'] = df['orig'] - df['floor']
    leftovers = df['rem'].sum().round().astype(int)
    if leftovers == 0:
        return (df['rem'] * 0)

    rows = df.sample(n=leftovers, weights=df['rem'], 
                     replace=False, random_state=random_state)
    rows['cts'] = 1
    df = df.join(rows['cts'].to_frame())
    
    return (df['cts'].fillna(0) + df['floor']).astype(int)