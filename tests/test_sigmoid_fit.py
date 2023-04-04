import numpy as np
import pytest

from talus_drc_helper.fit import fit_sigmoid, sigmoid


@pytest.fixture
def sample_curve():
    y_values = np.array(
        [
            202,
            678,
            2482,
            899,
            827,
            4444,
            79876,
            391022,
            505474,
            517617,
            500450,
            508352,
            419578,
            476165,
        ]
    )
    x_values = 100 / (2 ** np.arange(len(y_values))[::-1])

    return y_values, x_values


def test_fitting_drc_works(sample_curve):
    y_values, x_values = sample_curve
    popt, cs = fit_sigmoid(y_values, np.log(x_values))
    x = np.linspace(np.log(x_values).min(), np.log(x_values).max(), 1000)
    sigmoid(x, *popt)

    assert popt[1] > -2
    assert popt[1] < 2
    assert popt[0] > np.median(y_values)
    assert popt[0] < (1.2 * np.max(y_values))
