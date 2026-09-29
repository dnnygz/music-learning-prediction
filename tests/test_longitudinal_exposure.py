from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.measurement.longitudinal_exposure import build_daily_exposure


class LongitudinalExposureTests(unittest.TestCase):
    def test_lag_and_between_mean_use_training_days_only(self) -> None:
        events = pd.DataFrame({
            "user_id": ["student", "student", "student"],
            "day_index": [0, 20, 21],
            "total_evaluated": [9, 99, 999],
        })
        exposure = build_daily_exposure(events).set_index("day_index")

        self.assertEqual(exposure.loc[0, "exposure_raw"], 0)
        self.assertAlmostEqual(exposure.loc[1, "exposure_raw"], np.log1p(9))
        self.assertAlmostEqual(exposure.loc[21, "exposure_raw"], np.log1p(99))
        self.assertAlmostEqual(exposure.loc[22, "exposure_raw"], np.log1p(999))
        self.assertEqual(exposure.between_exposure.nunique(), 1)
        # Day-20 activity becomes the lag for day 21, outside the training mean.
        expected_training_mean = np.log1p(9) / 21
        self.assertAlmostEqual(
            exposure.loc[22, "between_exposure"], expected_training_mean
        )


if __name__ == "__main__":
    unittest.main()
