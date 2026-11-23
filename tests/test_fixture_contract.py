"""检查采集资料的身份、形状和可交叉引用性。"""

import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_fixture(name):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


class FixtureContractTests(unittest.TestCase):
    def test_record_manifest(self):
        manifest = read_fixture("manifest.json")
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        self.assertEqual(manifest["fixture"], "stereo_volume")
        self.assertEqual(manifest["records"], ["primary.json", "revision.json"])
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(revision["replaces"], primary["capture"])
        self.assertNotEqual(revision["capture"], primary["capture"])

    def test_related_input_records(self):
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        height, width = primary["camera"]["shape"]
        pixels = {}
        for tile in primary["tiles"] + revision["arrivals"]:
            y, x = tile["origin"]
            values = np.asarray(tile["features"], dtype=np.float64)
            self.assertEqual(values.ndim, 3)
            self.assertLessEqual(y + values.shape[0], height)
            self.assertLessEqual(x + values.shape[1], width)
            for dy, dx in np.ndindex(values.shape[:2]):
                key = (tile["side"], tile["calibration"], y + dy, x + dx)
                value = tuple(values[dy, dx])
                self.assertEqual(pixels.setdefault(key, value), value)
        for window in primary["requests"]:
            self.assertEqual(window["calibration"], primary["calibration"])
