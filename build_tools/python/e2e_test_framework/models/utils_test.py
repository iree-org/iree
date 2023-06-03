## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import string
import unittest

from e2e_test_framework.definitions import common_definitions
import e2e_test_framework.models.utils as model_utils


class UtilsTest(unittest.TestCase):

  def test_partial_template_substitute(self):
    template = string.Template("${name}-${batch_size}")

    result = model_utils.partial_template_substitute(template, name="xyz")

    self.assertEqual(result.substitute(batch_size=10), "xyz-10")

  def test_generate_batch_models(self):
    models = model_utils.generate_batch_models(
        id_template=string.Template("1234-${batch_size}"),
        name_template=string.Template("model-batch-${batch_size}"),
        tags=["abc"],
        source_url_template=string.Template(
            "https://example.com/x/${batch_size}.mlir"),
        source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
        entry_function="forward",
        input_type_templates=[
            string.Template("${batch_size}x128"),
            string.Template("${batch_size}x256")
        ],
        batch_sizes=[1, 4])

    self.assertEqual(
        models, {
            1:
                common_definitions.Model(
                    id="1234-1",
                    name="model-batch-1",
                    tags=["abc", "batch-1"],
                    source_url="https://example.com/x/1.mlir",
                    source_type=common_definitions.ModelSourceType.
                    EXPORTED_STABLEHLO_MLIR,
                    entry_function="forward",
                    input_types=["1x128", "1x256"]),
            4:
                common_definitions.Model(
                    id="1234-4",
                    name="model-batch-4",
                    tags=["abc", "batch-4"],
                    source_url="https://example.com/x/4.mlir",
                    source_type=common_definitions.ModelSourceType.
                    EXPORTED_STABLEHLO_MLIR,
                    entry_function="forward",
                    input_types=["4x128", "4x256"])
        })

  def test_generate_batch_models_missing_substitution(self):
    id_template_with_unknown = string.Template("1234-${unknown}-${batch_size}")

    self.assertRaises(
        KeyError, lambda: model_utils.generate_batch_models(
            id_template=id_template_with_unknown,
            name_template=string.Template("model-batch-${batch_size}"),
            tags=["abc"],
            source_url_template=string.Template(
                "https://example.com/x/${batch_size}.mlir"),
            source_type=common_definitions.ModelSourceType.
            EXPORTED_STABLEHLO_MLIR,
            entry_function="forward",
            input_type_templates=[
                string.Template("${batch_size}x128"),
            ],
            batch_sizes=[1, 4]))


if __name__ == "__main__":
  unittest.main()
