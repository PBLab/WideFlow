import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_rel


def least_sqaure_standard_error(x, y, a=None, b=None, a_h_null=0, b_h_null=0):
    """
    returns the standard error of a least sqaure estimation
    for a fitting model of the form: Y = a + b*x + e
    Args:
        x: 1D numpy array
        y: 1D numpy array

    Returns:

    """
    n = x.shape[0]

    if a is None or b is None:
        reg = LinearRegression().fit(x, y)
        a = reg.intercept_
        b = reg.coef_[0][0]

    y_pred = a + b * x

    x_mean = np.mean(x)
    x_div_sqr = np.sum(np.square(x - x_mean))

    y_mean = np.mean(y)
    y_div_sqr = np.sum(np.square(y - y_pred))

    SEb = np.sqrt(1 / (n - 2) * y_div_sqr) / np.sqrt(x_div_sqr)
    SEa = SEb * np.sqrt(1 / n * np.sum(np.square(x)))
    # residual =

    return SEa, SEb

