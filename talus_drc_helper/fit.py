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
    median_ys = np.median(ydata)
    median_x_data = np.median(xdata)
    p0 = [max_ys, median_x_data, 1, min_ys]  # this is an mandatory initial guess
    bounds = (
        (median_ys, np.min(xdata), -np.inf, 0),
        (max_ys, np.max(xdata), np.inf, median_ys),
    )
    popt, pcov = curve_fit(
        sigmoid, xdata, ydata, p0, method="dogbox", maxfev=5000, bounds=bounds
    )
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
        DRCEstimator(log_transform_x=True)
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

    def predict(self, X: NDArray) -> NDArray:  # noqa: N803
        """Predicts the values for the given X values.

        Parameters
        ----------
        X, NDArray[float]
            X values to pass to the function.
        """
        if self.log_transform_x:
            X = np.log(X)

        y = sigmoid(X, *self.popt)
        return y

    @property
    def parameters(self) -> dict[str, float]:
        """Returns the parameters of the fit.

        The parameters are returned as a dictionary
        with names that make sense on the context of a DRC fit.

        ┌─────────────────────────────────────┐
        │   Output Scaling   ->    ▗▄▞▀▀▀▀▀▀▀▀│
        │                     ▗▘              │ 400,000
        │                   ▞                 │ 300,000
        │                ▞ ^                  │ 200,000
        │             ▗▘   |  Inflection      │ 100,000
        │            ▞▘                       │
        │▄▄▄▄▄▄▄▞▀▘      <- Bias              │
        └─────────────────────────────────────┘

        Returns
        -------
        dict[str, float]
            A dictionary with the parameters of the fit.
            The named parameters are:
                - OutputScaling
                - Inflection
                - InputScaling
                - Bias
                - NormalizedResiduals

        """
        return {
            "OutputScaling": self.popt[0],
            "Inflection": self.popt[1],
            "InputScaling": self.popt[2],
            "Bias": self.popt[3],
            "NormalizedResiduals": self.residuals,
        }

    @property
    def ic50(self) -> float:
        """Returns the IC50 of the fit.

        Technically, the IC50 is the point where the output is 50% of the maximum
        Achievable effect by the drug.
        Which technically differs from the LD50, which is the point where the output
        is 50% of death is achieved.
        """
        if hasattr(self, "popt"):
            out = self.popt[1]
            if self.log_transform_x:
                out = np.exp(out)
            return out
        raise RuntimeError(
            "The DRC estimator has not been fit yet!, please use the fit method before"
            " you try to extract the IC50"
        )

    def ld_quantile(self, q: float | NDArray = 0.5) -> float:
        """Returns the LDXX of the fit.

        Parameters
        ----------
        q : float, optional
            The quantile to calculate, by default 0.5.
            q = 0.5 would calculate the LD50,
            q = 0.1 would calculate the LD10, etc.


        Returns
        -------
        float
            The LDXX of the fit for the specified value.
            It will return `nan` if the quantile is not
            in the range of the function.
        """
        x0 = self.popt[1]
        k = self.popt[2]

        x = (np.log((1 / q) - 1) / -k) + x0

        if self.log_transform_x:
            x = np.exp(x)
        return x

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

    def _plot(self, X, y):  # noqa
        raise NotImplementedError

    def terminal_plot(self, x: NDArray = None, y: NDArray = None) -> None:
        """Plots the fit in the terminal."""
        if x and y:
            pass
        else:
            x, y = self._sample_curve

        uniplot.plot(xs=x, ys=y, lines=True, x_as_log=self.log_transform_x)


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
    preds = my_est.predict(x_values)
