import logging
import sys
import unittest

import pytest
import torch
from iree.importers.torch.exporter import export

logger = logging.getLogger(__file__)


class NonStrictExportTest(unittest.TestCase):
    def testNonStrictExport(self):
        test_module = UnsupportedFunc()
        random_input = torch.randn((3, 3))
        exported = export(test_module, args=(random_input,), strict_export=False)
        mlir_str = str(exported.mlir_module)
        self.assertIn("func.func", mlir_str)

    def testStrictExportFailure(self):
        test_module = UnsupportedFunc()
        random_input = torch.randn((3, 3))
        with self.assertRaises(Exception):
            export(test_module, args=(random_input,), strict_export=True)


# Test module to check that aot strict_export works as intended. id() is a builin python
# function that will result in a graph break with export that uses torch dynamo (strict = True),
# but it will pass when using the Python interpreter (strict = False).
class UnsupportedFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        logger.warning("This will fail without disabling strict export mode")
        x = x + 1
        return x + id(x)


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)
