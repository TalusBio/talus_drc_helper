from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import uniplot
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator


# https://stackoverflow.com/a/62215374
# Solution from stack overflow
def sigmoid(
    x: NDArray[np.float64], L: float, x0: float, k: float, b: float
) -> NDArray[np.float64]:
    """Implements a sigmoid function.

    This function calculates the y value for every x value passed (as an array)

    Parameters
    ----------
    x, NDArray[float]
        X values to pass to the function.
    L, float
        see details
    x0, float
        see details
    k, float
        see details
    b, float
        see details

    Details
    -------

    L is responsible for scaling the output range from [0,1] to [0,L]
    x0 is the point in the middle of the Sigmoid
        i.e. the point where Sigmoid should originally output the value
        1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    k is responsible for scaling the input, which remains in (-inf,inf)
        Large values of K mean that the curve will have a much steeper increase.
    b adds bias to the output and changes its range from [0,L] to [b,L+b].
        In other words, moves the output up and down.


    """
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def fit_sigmoid(ydata: NDArray, xdata: NDArray) -> tuple[NDArray, NDArray]:
    """Gives the parameters for a sigmoid function.

    Returns
    -------
    popt
        Optimal calculated values
    pcov
        Estimated covariances
    """
    max_ys = max(ydata)
    min_ys = 0
    p0 = [max_ys, np.median(xdata), 1, min_ys]  # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method="dogbox")
    return popt, pcov


class DRCEstimator(BaseEstimator):
    """Implements a class to calculate DRC fits."""

    def __init__(self, log_transform_x: bool = False) -> None:
        """Starts an estimator.

        Usage
        -----
        >>> my_est = DRCEstimator(log_transform_x = True)
        >>> y_values = np.array( [ 202, 827, 4444, 79876, 391022, 508352, 476165, ])
        >>> x_values = 100 / (2 ** np.arange(len(y_values))[::-1])
        >>> my_est.fit(x_values, y_values)
        >>> # my_est.predict(x_values)
        """
        self.log_transform_x = log_transform_x

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.float64]  # noqa: N803
    ) -> DRCEstimator:
        """Fits the estimator.

        Parameters
        ----------
        X, NDArray[float]
            X values to pass to the function.
        y, NDArray[float]
            Y values to pass to the function.
        """
        if self.log_transform_x:
            X = np.log(X)  # noqa: N806
        self.popt, self.pcov = fit_sigmoid(ydata=y, xdata=X)
        self.residuals = (((y - sigmoid(X, *self.popt)) / self.popt[0]) ** 2).mean()

        return self

    def predict(self, X: NDArray) -> NDArray:
        if self.log_transform_x:
            X = np.log(X)

        y = sigmoid(X, *self.popt)
        return y

    @property
    def parameters(self) -> dict[str, float]:
        return {
            "OutputScaling": self.popt[0],
            "Inflection": self.popt[1],
            "InputScaling": self.popt[2],
            "Bias": self.popt[3],
            "NormalizedResiduals": self.residuals,
        }

    @property
    def ic50(self) -> float:
        if hasattr(self, "popt"):
            out = self.popt[1]
            if self.log_transform_x:
                out = np.exp(out)
            return out
        raise RuntimeError(
            "The DRC estimator has not been fit yet!, please use the fit method before"
            " you try to extract the IC50"
        )

    @property
    def _sample_xs(self) -> NDArray:
        side = 10 * 1 / self.popt[2]
        xs = np.linspace(start=self.popt[1] - side, stop=self.popt[1] + side, num=200)
        if self.log_transform_x:
            xs = np.exp(xs)
        return xs

    @property
    def _sample_curve(self) -> tuple[NDArray, NDArray]:
        x = self._sample_xs
        y = self.predict(x)
        return x, y

    @property
    def _sample_curve_df(self) -> pd.DataFrame:
        x, y = self._sample_curve
        return pd.DataFrame({"x": x, "y": y})

    @property
    def _sample_curve_plot(self) -> alt.Chart:
        scaling = alt.Scale(type="log" if self.log_transform_x else "linear")
        base_chart = (
            alt.Chart(self._sample_curve_df)
            .mark_line()
            .encode(alt.X("x", scale=scaling), alt.Y("y"))
        )
        return base_chart

    def plot(self, X, y):
        pass

    def terminal_plot(self, x=None, y=None) -> None:
        if x and y:
            pass
        else:
            x, y = self._sample_curve

        if self.log_transform_x:
            x = np.log(x)

        uniplot.plot(xs=x, ys=y, lines=True)


if __name__ == "__main__":
    my_est = DRCEstimator(log_transform_x=True)
    y_values = np.array(
        [
            202,
            827,
            4444,
            79876,
            391022,
            508352,
            476165,
        ]
    )
    x_values = 100 / (2 ** np.arange(len(y_values))[::-1])
    my_est.fit(x_values, y_values)
    my_est.terminal_plot()
    bc = my_est._sample_curve_plot
