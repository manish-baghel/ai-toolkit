"""Focused tests for throttled training telemetry.

Run directly::

    PYTHONPATH=. python -m unittest -v testing.test_training_reporting
"""

import unittest
from collections import OrderedDict

from toolkit.config_modules import LoggingConfig
from toolkit.progress_bar import is_progress_update_due, materialize_metrics


class _ScalarSpy:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def item(self):
        self.calls += 1
        return self.value


class TrainingReportingTest(unittest.TestCase):
    def test_progress_interval_defaults_to_twenty(self):
        self.assertEqual(LoggingConfig().progress_every, 20)

    def test_progress_interval_can_be_overridden(self):
        self.assertEqual(LoggingConfig(progress_every=7).progress_every, 7)

    def test_progress_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            LoggingConfig(progress_every=0)
        with self.assertRaises(ValueError):
            LoggingConfig(progress_every=-1)

    def test_progress_cadence_includes_first_interval_and_final_steps(self):
        due = [
            step
            for step in range(45)
            if is_progress_update_due(step, 0, 45, 20)
        ]
        self.assertEqual(due, [0, 19, 39, 44])

    def test_resumed_progress_cadence_starts_immediately(self):
        due = [
            step
            for step in range(37, 45)
            if is_progress_update_due(step, 37, 45, 20)
        ]
        self.assertEqual(due, [37, 39, 44])

    def test_metrics_are_materialized_once_only_when_consumed(self):
        loss = _ScalarSpy(0.25)
        metrics = OrderedDict(loss=loss)

        for step in range(45):
            if is_progress_update_due(step, 0, 45, 20):
                materialized = materialize_metrics(metrics)
                self.assertEqual(materialized['loss'], 0.25)

        self.assertEqual(loss.calls, 4)


if __name__ == '__main__':
    unittest.main()
