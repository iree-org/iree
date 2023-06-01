## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities to help define models."""

import string
from typing import Dict, Sequence

from e2e_test_framework.definitions import common_definitions


def generate_batch_models(
    id_template: str, name_template: str, tags: Sequence[str],
    source_type: common_definitions.ModelSourceType, source_url_template: str,
    entry_function: str, input_type_templates: Sequence[str],
    batch_sizes: Sequence[int]) -> Dict[int, common_definitions.Model]:
  """Generate model definitions for different batch sizes by substituting
  ${BATCH_SIZE}` in the template strings.

  Only `*_template` parameters will be treated as templates and substituted. A
  `batch-<batch size>` tag will be appended to the tags in each returned model.

  Returns:
    Map of batch size to model.
  """
  model_map = {}
  for batch_size in batch_sizes:
    substituted_input_types = [
        string.Template(input_type).substitute(BATCH_SIZE=batch_size)
        for input_type in input_type_templates
    ]
    model_map[batch_size] = common_definitions.Model(
        id=string.Template(id_template).substitute(BATCH_SIZE=batch_size),
        name=string.Template(name_template).substitute(BATCH_SIZE=batch_size),
        tags=list(tags) + [f"batch-{batch_size}"],
        source_type=source_type,
        source_url=string.Template(source_url_template).substitute(
            BATCH_SIZE=batch_size),
        entry_function=entry_function,
        input_types=substituted_input_types)

  return model_map
