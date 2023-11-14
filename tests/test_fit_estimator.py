import numpy as np

from talus_drc_helper.fit import DRCEstimator


def test_estimator_scale():
    """Tests that the estimator scales the data correctly.

    at fit and inference time.
    """
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

    # The LD50 should be in the range of the x values.
    assert my_est.popt[1] > 0
    assert my_est.popt[1] < np.max(x_values)
    assert my_est.popt[1] < 10

    # This makes sure the predictions are in the same scale as the
    # original data.
    assert np.max(preds) / np.max(y_values) < 1.1
    assert np.max(preds) / np.max(y_values) > 0.8
