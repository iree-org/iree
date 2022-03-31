# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Updates the TFLite integration test model documentation.

When changes are made in the ../integrations/tensorflow/test/iree_tfl_tests 
directory, please run this script to update the table of models.
"""

import os
import utils
from pathlib import Path


def main():
  current_dir = os.path.dirname(__file__)
  model_dir = os.path.join(current_dir,
                           '../integrations/tensorflow/test/iree_tfl_tests')

  files = list(Path(model_dir).glob('*.run'))
  models = [[0 for x in range(2)] for y in range(len(files))]

  for i in range(len(files)):
    name = os.path.basename(files[i].name).replace('.run', '')
    models[i][0] = name.ljust(20)

    status = 'PASS'
    with open(files[i], 'r') as file:
      if 'XFAIL' in file.read():
        status = 'XFAIL'
    models[i][1] = status

  with open(os.path.join(model_dir, 'README.md'),
            'w') as tflite_model_documentation:
    tflite_model_documentation.write('# TFLite integration tests status\n\n' \
    'This documentation shows the models that are currently being tested on IREE\'s\n' \
    'presubmits.  If any tests are added or changed, please run\n' \
    'scripts/update_tflite_model_documentation.py to update this table.\n\n' \
    '|       Model        |      Status        |\n' \
    '| ------------------ | ------------------ |\n')
    tflite_model_documentation.write(utils.create_markdown_table(models))


if __name__ == '__main__':
  main()
  