"""The adapter `cosyvoice_dpo` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "cosyvoice_dpo"


if __name__ == "__main__":
    unittest.main()
