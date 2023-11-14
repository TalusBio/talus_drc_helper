from __future__ import annotations

import warnings
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import uniplot
from loguru import logger
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
    # TODO add an option to normalize the data prior to fitting
    # it seems like the fit is much better when data is in similar ranges

    max_ys = max(ydata)
    min_ys = 0
    median_ys = np.median(ydata)
    median_x_data = np.median(xdata)

    # Check if the data is increasing or decreasing
    data_decreasing = True
    if ydata[xdata < median_x_data].mean() > ydata[xdata > median_x_data].mean():
        # logger.debug("Data is increasing, flipping the slope")
        data_decreasing = False

    prior_slope = 1 if data_decreasing else -1
    prior_range = max_ys - min(ydata)

    p0 = [
        prior_range,
        median_x_data,
        prior_slope,
        min_ys,
    ]  # this is an mandatory initial guess
    bias_lower_bound = 0
    bias_upper_bound = median_ys
    lower_bound_L = prior_range / 2
    higher_bound_L = 2 * prior_range
    if np.any(ydata < 0):
        warnings.warn("Some Y values are negative, this may cause issues with the fit")
        # This is OK if we are calculating GR50s
        bias_lower_bound = -1
        bias_upper_bound = 2
        p0[-1] = -0.99
        lower_bound_L = 0.1
        higher_bound_L = 2
        p0[0] = min(p0[0], 2)

    bounds = (
        (lower_bound_L, np.min(xdata), -np.inf, bias_lower_bound),
        (higher_bound_L, np.max(xdata), np.inf, bias_upper_bound),
    )

    for p, b, bu in zip(p0, *bounds):
        if (p < b) or (p > bu):
            raise ValueError(
                f"Initial guess {p} is not within the bounds {b} < {p} < {bu}"
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
            X = np.log10(X)  # noqa: N806

        non_inf_non_missing_X = np.isfinite(X)
        if np.any(non_inf_non_missing_X):
            warnings.warn(
                "Some X values are infinite or missing. They will be ignored."
            )
        self.popt, self.pcov = fit_sigmoid(
            ydata=y[non_inf_non_missing_X], xdata=X[non_inf_non_missing_X]
        )
        self.residuals = np.sqrt(
            (((y - sigmoid(X, *self.popt)) / self.popt[0]) ** 2).mean()
        )

        return self

    def predict(self, X: NDArray) -> NDArray:  # noqa: N803
        """Predicts the values for the given X values.

        Parameters
        ----------
        X, NDArray[float]
            X values to pass to the function.
        """
        if self.log_transform_x:
            X = np.log10(X)

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

    def report_df(self):
        """Returns a pandas dataframe with the parameters of the fit.

        Returns
        -------
        pd.DataFrame
            A dataframe with the parameters of the fit.
            The named parameters are:
                - OutputScaling
                - Inflection
                - InputScaling
                - Bias
                - NormalizedResiduals
                - AbsIC_50
                - AbsIC_5
                - AbsIC_95
        """
        params = self.parameters
        params["Log10Inflection"] = params["Inflection"]
        params["AbsIC_5"] = self.ld_quantile(1 - 0.05)
        params["AbsIC_50"] = self.ld_quantile(1 - 0.5)
        params["AbsIC_95"] = self.ld_quantile(1 - 0.95)
        if self.log_transform_x:
            for key in ["Inflection"]:
                params[key] = 10 ** params[key]

        return pd.DataFrame([params])

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
                out = 10**out
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
        L = self.popt[0]
        b = self.popt[3]
        if L > 3:
            # L == 2 is normal for GR50 fits
            # L == 1 is normal for LD50 fits
            logger.warning(
                f"The output scaling is greater than 3 {L}, this may be an issue"
            )

        x = (np.log((L / (q - b)) - 1) / -k) + x0

        if self.log_transform_x:
            x = 10**x
        return x

    @property
    def _sample_xs(self) -> NDArray:
        side = 10 * 1 / self.popt[2]
        xs = np.linspace(start=self.popt[1] - side, stop=self.popt[1] + side, num=200)
        if self.log_transform_x:
            xs = 10**xs
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

    def _plot(self, target_file, title, X, y, ax=None):  # noqa
        passed_ax = ax is not None
        if not passed_ax:
            import vizta
            from matplotlib import pyplot as plt

            vizta.mpl.set_theme()
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        _x, _y = self._sample_curve
        ax.plot(_x, _y, label=title)
        ax.scatter(X, y, label=title)
        hlines_show = [0, 0.5, 1]
        if np.any(y < 0):
            hlines_show.append(-1)

        ax.hlines(
            y=hlines_show,
            xmin=min(_x),
            xmax=max(_x),
            linestyles="dashed",
            color="gray",
            alpha=0.3,
        )
        ax.set_xscale("log")

        if not passed_ax:
            ax.set_title(title)
            if target_file is not None:
                fig.savefig(target_file, bbox_inches="tight", dpi=175)
            plt.show()

    def terminal_plot(self, x: NDArray = None, y: NDArray = None) -> None:
        """Plots the fit in the terminal."""
        if x and y:
            pass
        else:
            x, y = self._sample_curve

        uniplot.plot(xs=x, ys=y, lines=True, x_as_log=self.log_transform_x)


@dataclass
class DRCEstimatorGroup:
    estimators: list[DRCEstimator]
    grouping_variables: list[str]
    groupings: tuple[tuple[str]]
    target_variable: str
    dose_variable: str

    def __post_init__(self):
        if len(self.groupings) != len(self.estimators):
            raise ValueError(
                "The number of estimators and groupings should be the same"
            )
        if len(self.grouping_variables) != len(self.groupings[0]):
            raise ValueError(
                "The number of grouping variables should be the same as the number of"
                " groupings"
            )
        if isinstance(self.groupings[0], str):
            self.groupings = tuple([(g,) for g in self.groupings])

    @property
    def _sample_curve_df(self) -> pd.DataFrame:
        to_concat = []
        mapper = {"x": self.dose_variable, "y": self.target_variable}
        for est, grouping in zip(self.estimators, self.groupings):
            df = est._sample_curve_df
            for g, v in zip(grouping, self.grouping_variables):
                df[v] = g

            df.rename(columns=mapper, inplace=True)
            to_concat.append(df)
        out = pd.concat(to_concat)
        return out

    def report_df(self) -> pd.DataFrame:
        to_concat = []
        for est, grouping in zip(self.estimators, self.groupings):
            df = est.report_df()
            for g, v in zip(grouping, self.grouping_variables):
                df[v] = g
            to_concat.append(df)
        out = pd.concat(to_concat)
        # Reorder the columns
        out = out[
            list(self.grouping_variables)
            + list(out.columns.difference(self.grouping_variables))
        ]
        return out


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
