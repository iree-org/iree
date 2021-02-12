# Copyright 2021 Google LLC
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

import copy
from typing import Any

import pyiree as iree
import pyiree.test

from absl import logging
from absl.testing import absltest
import numpy as np

try:
  import jax
  JAX_IS_AVALIABLE = True
except ModuleNotFoundError:
  JAX_IS_AVALIABLE = False

# pytype thinks pyiree.jax is jax.
# pytype: disable=module-attr

INPUT_TREE = {
    "z": [1, (2,), {
        "y": 3,
        "x": {
            "w": (None, 3.14, -46)
        }
    }],
    "v": {
        "u": 99,
        "t": np.array([100])
    }
}
LEAVES = [1, 2, 3, None, 3.14, -46, 99, np.array([100])]
LEAVES_SORTED = [np.array([100]), 99, 1, 2, None, 3.14, -46, 3]


class TreeUtilsTest(absltest.TestCase):

  def _assert_ordered_equal(self, first: Any, second: Any):
    if isinstance(first, dict):
      self._assert_ordered_equal(list(first.keys()), list(second.keys()))
      self._assert_ordered_equal(list(first.values()), list(second.values()))
    else:
      self.assertEqual(first, second)

  def test_inverse(self):
    input_tree = copy.deepcopy(INPUT_TREE)
    output_tree = iree.test.tree_unflatten(
        *iree.test.tree_flatten(input_tree, sort_keys=False))
    self._assert_ordered_equal(input_tree, output_tree)

  def test_not_inverse(self):
    input_tree = copy.deepcopy(INPUT_TREE)
    output_tree = iree.test.tree_unflatten(*iree.test.tree_flatten(input_tree))
    self.assertEqual(input_tree, output_tree)
    with self.assertRaises(AssertionError):
      self._assert_ordered_equal(input_tree, output_tree)

  def test_leaves_unsorted(self):
    input_tree = copy.deepcopy(INPUT_TREE)
    leaves, _ = iree.test.tree_flatten(input_tree, sort_keys=False)
    self.assertEqual(LEAVES, leaves)

  def test_leaves_sorted(self):
    input_tree = copy.deepcopy(INPUT_TREE)
    leaves, _ = iree.test.tree_flatten(input_tree)
    self.assertEqual(LEAVES_SORTED, leaves)

  def test_not_in_place(self):
    input_tree = copy.deepcopy(INPUT_TREE)

    leaves, tree_def = iree.test.tree_flatten(input_tree)
    leaves[0][0] = -100
    self.assertNotEqual(input_tree["v"]["t"], leaves[0])

    output_tree = iree.test.tree_unflatten(leaves, tree_def)
    self.assertNotEqual(input_tree, output_tree)
    output_tree["v"]["t"][0] = 33
    self.assertNotEqual(output_tree["v"]["t"], leaves[0])

  def test_in_place(self):
    input_tree = copy.deepcopy(INPUT_TREE)

    leaves, tree_def = iree.test.tree_flatten(input_tree, in_place=True)
    leaves[0][0] = -100
    self.assertEqual(input_tree["v"]["t"], leaves[0])

    output_tree = iree.test.tree_unflatten(leaves, tree_def, in_place=True)
    output_tree["v"]["t"][0] = 33
    self.assertEqual(input_tree, output_tree)

  def test_tree_map(self):
    cube = lambda x: x**3 if x is not None else None
    input_tree = copy.deepcopy(INPUT_TREE)

    output_tree = iree.test.tree_map(cube, input_tree)
    output_leaves, output_tree_def = iree.test.tree_flatten(output_tree)
    self.assertEqual(list(map(cube, LEAVES_SORTED)), output_leaves)
    _, input_tree_def = iree.test.tree_flatten(input_tree)
    self.assertEqual(input_tree_def, output_tree_def)

  def test_matches_xla_sort_order(self):
    if not JAX_IS_AVALIABLE:
      logging.warning("JAX not installed, skippig this unit test.")
      return

    input_tree = copy.deepcopy(INPUT_TREE)
    output_tree = iree.test.tree_unflatten(*iree.test.tree_flatten(input_tree))
    self.assertEqual(input_tree, output_tree)

    xla_values, xla_tree = jax.tree_flatten(input_tree)
    xla_output_tree = jax.tree_unflatten(xla_tree, xla_values)
    self._assert_ordered_equal(output_tree, xla_output_tree)

  def test_tuples_are_leaves(self):

    def tuple_is_leaf(tree):
      return isinstance(tree, tuple) or iree.test.is_leaf(tree)

    input_tree = copy.deepcopy(INPUT_TREE)
    expected_leaves = [np.array([100]), 99, 1, (2,), (None, 3.14, -46), 3]
    leaves, tree_def = iree.test.tree_flatten(input_tree, tuple_is_leaf)
    self.assertEqual(leaves, expected_leaves)

  def test_base_cases(self):
    round_trip_none = iree.test.tree_unflatten(*iree.test.tree_flatten(None))
    self.assertEqual(round_trip_none, None)

    round_trip_one = iree.test.tree_unflatten(*iree.test.tree_flatten(1))
    self.assertEqual(round_trip_one, 1)


if __name__ == "__main__":
  absltest.main()
