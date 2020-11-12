#!/usr/bin/env python3

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Updates coverage of TensorFlow e2e tests on all backends.

Example usage: python3 update_e2e_coverage.py build-docs
"""

import argparse
import collections
import os
import re
import subprocess

import utils

TENSORFLOW_COVERAGE_DIR = 'tensorflow_coverage'
REFERENCE_BACKEND = 'tf'
# Assumes that tests are expanded for the tf, iree_vmla, iree_llvmjit and
# iree_vulkan backends.
BACKENDS_TO_TITLES = collections.OrderedDict([
    ('tf', 'tensorflow'),
    ('tflite', 'tflite'),
    ('iree_vmla', 'vmla'),
    ('iree_llvmjit', 'llvm-ir'),
    ('iree_vulkan', 'vulkan-spirv'),
])

KWS_LINK = (
    'https://github.com/google-research/google-research/tree/master/kws_streaming'
)
KWS_LINK = f'[Keyword Spotting Streaming]({KWS_LINK})'

COVERAGE_GROUP_TO_TEST_SUITES = {
    'tf_base_coverage': [
        '//integrations/tensorflow/e2e:e2e_tests',
        '//integrations/tensorflow/e2e/math:math_tests',
        '//integrations/tensorflow/e2e/math:math_dynamic_dims_tests',
        '//integrations/tensorflow/e2e/math:math_complex_tests',
    ],
    'tf_keras_coverage': [
        '//integrations/tensorflow/e2e/keras/layers:layers_tests',
        '//integrations/tensorflow/e2e/keras/layers:layers_dynamic_batch_tests',
        '//integrations/tensorflow/e2e/keras/layers:layers_training_tests',
    ],
    'language_and_speech_coverage': [
        '//integrations/tensorflow/e2e:mobile_bert_squad_tests',
        '//integrations/tensorflow/e2e/keras:keyword_spotting_tests',
        '//integrations/tensorflow/e2e/keras:keyword_spotting_internal_streaming_tests',
    ],
    'vision_coverage': [
        '//integrations/tensorflow/e2e/keras:imagenet_non_hermetic_tests',
        '//integrations/tensorflow/e2e/slim_vision_models:slim_vision_tests',
    ],
}

COVERAGE_GROUP_TO_TITLE = {
    'tf_base_coverage': 'TensorFlow Base APIs',
    'tf_keras_coverage': 'TensorFlow Keras Layers',
    'language_and_speech_coverage': 'Language and Speech Models',
    'vision_coverage': 'Vision Models',
}

COVERAGE_GROUP_TO_DESCRIPTION = {
    'tf_base_coverage':
        ('Tests of the `tf`, `tf.math`, `tf.nn`, `tf.signal` and `tf.strings` '
         'APIs.'),
    'tf_keras_coverage':
        ('Tests of `tf.keras.layers` compiled with static shapes, dynamic '
         'shapes and training enabled.'),
    'language_and_speech_coverage':
        'Tests of MobileBert and streamable Keyword Spotting models.',
    'vision_coverage':
        'Tests of Keras and Slim vision models.',
}

TEST_SUITES_TO_HEADERS = {
    # tf_base_coverage
    '//integrations/tensorflow/e2e:e2e_tests':
        'End to end TensorFlow tests',
    '//integrations/tensorflow/e2e/math:math_tests':
        'End to end tests of tf.math functions with static dimensions',
    '//integrations/tensorflow/e2e/math:math_dynamic_dims_tests':
        'End to end tests of tf.math functions with dynamic dimensions',
    '//integrations/tensorflow/e2e/math:math_complex_tests':
        'End to end tests of tf.math functions with complex numbers',
    # tf_keras_coverage
    '//integrations/tensorflow/e2e/keras/layers:layers_tests':
        'End to end tests of tf.keras layers (with default configuration and '
        'static batch sizes in inference mode)',
    '//integrations/tensorflow/e2e/keras/layers:layers_full_api_tests':
        'End to end tests of tf.keras layers full APIs '
        '(with static batch sizes in inference mode)',
    '//integrations/tensorflow/e2e/keras/layers:layers_dynamic_batch_tests':
        'End to end tests of tf.keras layers with dynamic batch sizes '
        '(with default configuration in inference mode)',
    '//integrations/tensorflow/e2e/keras/layers:layers_training_tests':
        'End to end tests of tf.keras layers in training mode (with default'
        'configuration and static batch sizes)',
    # language_and_speech_coverage
    '//integrations/tensorflow/e2e:mobile_bert_squad_tests':
        'End to end test of MobileBert on SQuAD',
    '//integrations/tensorflow/e2e/keras:keyword_spotting_tests':
        f'End to end tests of {KWS_LINK} models',
    '//integrations/tensorflow/e2e/keras:keyword_spotting_internal_streaming_tests':
        f'End to end tests of {KWS_LINK} models in internal streaming mode',
    # vision_coverage
    '//integrations/tensorflow/e2e/keras:imagenet_non_hermetic_tests':
        'End to end tests of tf.keras.applications vision models on Imagenet',
    '//integrations/tensorflow/e2e/slim_vision_models:slim_vision_tests':
        'End to end tests of TensorFlow slim vision models',
}

TEST_SUITES_TO_NOTES = {
    '//integrations/tensorflow/e2e/math:math_tests': (
        '**Note:** To be thorough, these tests use high rank tensors and\n'
        'test int dtypes where TensorFlow allows them to be used. Both of\n'
        'these choices disproportionately affect TFLite coverage, and\n'
        'don\'t represent coverage for simple use cases.\n'),
    '//integrations/tensorflow/e2e/keras/layers:layers_tests': (
        '**Note:** Layers like `Dropout` are listed as passing in this table,\n'
        'but they function similar to identity layers in these tests. **See \n'
        'the third table for the coverage of these layers during training.**\n'
        '\n'
        'These tests also only modify required `tf.keras.layers` arguments.\n'
        'See the full API tests below for coverage on of non-default\n'
        'layer configurations.'),
}
# Key to use as the name of the rows in the left column for each test in the
# suite.
TEST_SUITE_TO_ROW_ID_KEY = {
    # tf_base_coverage
    '//integrations/tensorflow/e2e/math:math_tests':
        'functions',
    '//integrations/tensorflow/e2e/math:math_dynamic_dims_tests':
        'functions',
    '//integrations/tensorflow/e2e/math:math_complex_tests':
        'functions',
    # tf_keras_coverage
    '//integrations/tensorflow/e2e/keras/layers:layers_tests':
        'layer',
    '//integrations/tensorflow/e2e/keras/layers:layers_full_api_tests':
        'layer',
    '//integrations/tensorflow/e2e/keras/layers:layers_dynamic_batch_tests':
        'layer',
    '//integrations/tensorflow/e2e/keras/layers:layers_training_tests':
        'layer',
    # language_and_speech_coverage
    '//integrations/tensorflow/e2e/keras:keyword_spotting_tests':
        'model',
    '//integrations/tensorflow/e2e/keras:keyword_spotting_internal_streaming_tests':
        'model',
    # vision_coverage
    '//integrations/tensorflow/e2e/keras:imagenet_non_hermetic_tests':
        'model',
    '//integrations/tensorflow/e2e/slim_vision_models:slim_vision_tests':
        'model',
}

# Some test suites are generated from a single source. This allows us to point
# to the right test file when generating test URLs.
SINGLE_SOURCE_SUITES = {
    # tf_base_coverage
    '//integrations/tensorflow/e2e/math:math_tests':
        'math_test',
    '//integrations/tensorflow/e2e/math:math_dynamic_dims_tests':
        'math_test',
    '//integrations/tensorflow/e2e/math:math_complex_tests':
        'math_test',
    # tf_keras_coverage
    '//integrations/tensorflow/e2e/keras/layers:layers_tests':
        'layers_test',
    '//integrations/tensorflow/e2e/keras/layers:layers_full_api_tests':
        'layers_test',
    '//integrations/tensorflow/e2e/keras/layers:layers_dynamic_batch_tests':
        'layers_test',
    '//integrations/tensorflow/e2e/keras/layers:layers_training_tests':
        'layers_test',
    # language_and_speech_coverage
    '//integrations/tensorflow/e2e/keras:keyword_spotting_tests':
        'keyword_spotting_streaming_test',
    '//integrations/tensorflow/e2e/keras:keyword_spotting_internal_streaming_tests':
        'keyword_spotting_streaming_test',
    # vision_coverage
    '//integrations/tensorflow/e2e/keras:imagenet_non_hermetic_tests':
        'vision_model_test',
    '//integrations/tensorflow/e2e/slim_vision_models:slim_vision_tests':
        'slim_vision_model_test',
}

TARGET_EXCLUSION_FILTERS = [
    r'mobilenet_v1_.*',  # Slim vision MobileNetV1.
    r'mobilenet_v2_.*',  # Slim vision MobileNetV2.
]

# The symbols to show in the table if the operation is supported or not.
SUCCESS_ELEMENT = '<span class="success-table-element">✓</span>'
FAILURE_ELEMENT = '<span class="failure-table-element">✗</span>'

MAIN_URL = 'https://github.com/google/iree/tree/main'
TARGETS_URL = os.path.join(MAIN_URL, 'iree/compiler/Dialect/HAL/Target')

BACKEND_INFO = f"""IREE has three backend
[targets]({TARGETS_URL}):
`vmla`, `llvm-ir` and `vulkan-spirv`. We also test TFLite in our infrastructure
for benchmarking purposes. The coverage tables below are automatically generated
from IREE's test suites."""


def parse_arguments():
  """Parses command-line options."""
  parser = argparse.ArgumentParser(
      description='Generates Markdown files for op coverage table')
  parser.add_argument('build_dir',
                      metavar='BUILD_PATH',
                      type=str,
                      help='Base build directory.')

  parsed_args = parser.parse_args()
  if not os.path.isdir(parsed_args.build_dir):
    raise parser.error('expected path to a directory')

  return parsed_args


def parse_test_name(test_name, test_suite):
  """Splits a test name into a dictionary with its source file and backend."""
  test_name_parts = test_name.split("__")
  test_info = {}

  # The iree_e2e_test_suite elides a 'src' key before the name of the test
  # for brevity.
  if len(test_name_parts) % 2 == 1:
    test_info['src'] = test_name_parts.pop(0)

  # The rest of the test name should follow 'key__value__key__value__...'.
  for key, value in zip(test_name_parts[::2], test_name_parts[1::2]):
    test_info[key] = value

  # Default to using the test source file name as the row id for the table.
  if 'src' in test_info:
    test_info['row_id'] = test_info['src']
  else:
    test_info['src'] = SINGLE_SOURCE_SUITES[test_suite]
    test_info['row_id'] = test_info[TEST_SUITE_TO_ROW_ID_KEY[test_suite]]

  if 'target_backends' not in test_info:
    raise ValueError('Expected `target_backends` to be in the test name but '
                     f'got `{test_name}`.')

  return test_info


def get_name_and_backend(test_string):
  """Splits a pathless test target into its name and comparison backend."""
  name, backend = test_string.split(f'__{REFERENCE_BACKEND}__')
  return name, backend


def get_suite_metadata(test_suite):
  """Gets all test names, and passing and failing test-backend pairs."""
  passing = utils.get_test_targets(test_suite)
  failing = utils.get_test_targets(f'{test_suite}_failing')

  # Remove bazel path.
  passing = [test.replace(f'{test_suite}__', '') for test in passing]
  failing = [test.replace(f'{test_suite}_failing__', '') for test in failing]

  # Split into a dictionary mapping 'src', 'target_backend', ... to the
  # appropriate values for each test target.
  passing_info = [parse_test_name(test, test_suite) for test in passing]
  failing_info = [parse_test_name(test, test_suite) for test in failing]
  return passing_info, failing_info


def get_row_hyperlink(test_suite, row_id, test_source):
  """Returns a Markdown hyperlink pointing to the test source on GitHub."""
  # Convert `//path/to/tests:test_suite` to `path/to/tests`
  test_path = test_suite.replace('//', '').split(':')[0]

  test_url = os.path.join(MAIN_URL, test_path, f'{test_source}.py')
  return f'[{row_id}]({test_url})'


def generate_table(test_suite):
  """Generates an e2e backend coverage Markdown table."""
  passing_info, _ = get_suite_metadata(test_suite)

  # Create a dictionary mapping row names to source file names.
  row_id_to_source = {}
  for test_info in passing_info:
    row_id_to_source[test_info['row_id']] = test_info['src']

  # Create a dictionary mapping test names to a list of bools representing their
  # backend coverage.
  table = collections.defaultdict(lambda: [False] * len(BACKENDS_TO_TITLES))
  ordered_backends = list(BACKENDS_TO_TITLES.keys())
  for test_info in passing_info:
    backend_index = ordered_backends.index(test_info['target_backends'])
    table[test_info['row_id']][backend_index] = True

  # Create a header for the coverage table.
  reference_index = ordered_backends.index(REFERENCE_BACKEND)
  ordered_backend_titles = list(BACKENDS_TO_TITLES.values())
  # Remove the reference backend from the table header.
  ordered_backend_titles.pop(reference_index)
  first_row = ['target'] + ordered_backend_titles
  second_row = [':-:' for _ in first_row]

  # Generate the coverage table as a 2D array.
  rows = [first_row, second_row]
  for row_id, backends in sorted(table.items()):
    # If the reference backend is failing then there is no reason to show the
    # coverage of the other backends.
    if not backends[ordered_backends.index(REFERENCE_BACKEND)]:
      continue
    # Remove the reference backend from the row now that we know it's passing.
    backends.pop(reference_index)

    # Skip any rows defined in the TARGET_EXCLUSION_FILTERS.
    if any(re.match(pattern, row_id) for pattern in TARGET_EXCLUSION_FILTERS):
      continue

    row = [get_row_hyperlink(test_suite, row_id, row_id_to_source[row_id])]
    row.extend([
        SUCCESS_ELEMENT if backend else FAILURE_ELEMENT for backend in backends
    ])
    rows.append(row)
  return utils.create_markdown_table(rows)


def generate_coverage_doc(coverage_group, coverage_dir):
  paragraphs = [
      f'# {COVERAGE_GROUP_TO_TITLE[coverage_group]}',
      COVERAGE_GROUP_TO_DESCRIPTION[coverage_group],
      BACKEND_INFO,
  ]
  header = '\n\n'.join(paragraphs) + '\n\n'

  content = []
  for test_suite in COVERAGE_GROUP_TO_TEST_SUITES[coverage_group]:
    content.append(f'## {TEST_SUITES_TO_HEADERS[test_suite]}')
    if test_suite in TEST_SUITES_TO_NOTES:
      content.append(TEST_SUITES_TO_NOTES[test_suite])

    content.append(generate_table(test_suite))
  content = '\n\n'.join(content) + '\n'  # Trailing newline.

  table_path = os.path.join(coverage_dir, f'{coverage_group}.md')
  with open(table_path, 'w', encoding='utf-8') as f:
    f.write(header)
    f.write(content)


if __name__ == '__main__':
  args = parse_arguments()
  coverage_dir = os.path.join(args.build_dir, 'doc', TENSORFLOW_COVERAGE_DIR)
  os.makedirs(coverage_dir, exist_ok=True)

  for coverage_group in COVERAGE_GROUP_TO_TEST_SUITES:
    generate_coverage_doc(coverage_group, coverage_dir)
    print()
