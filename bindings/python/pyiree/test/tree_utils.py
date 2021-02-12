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
"""A minimal implementation of XLA's pytree flattening and unflattening."""

# TODO(#4131) python>=3.7: Use postponed type annotations.

import copy
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

__all__ = [
    "is_leaf",
    "tree_flatten",
    "tree_map",
    "tree_unflatten",
]

Leaf = Any
Branch = Union[Sequence["Tree"], Mapping[Any, "Tree"]]
Tree = Union[Leaf, Branch]  # pytype: disable=not-supported-yet

TreeDefLeaf = None
TreeDefBranch = Union[List["TreeDef"], Tuple["TreeDef"], Dict[Any, "TreeDef"]]
TreeDef = Union[TreeDefLeaf, TreeDefBranch]  # pytype: disable=not-supported-yet


def is_leaf(tree: Tree) -> bool:
  return not isinstance(tree, (Sequence, Mapping))


def _set_element(tree: TreeDefBranch, index: Any, value: Tree) -> Branch:
  """Abstracts setting (or appending) an element to a list, tuple or dict."""
  if isinstance(tree, tuple):
    tree = tree[:index] + (value,) + tree[index + 1:]
  elif isinstance(tree, dict) or index < len(tree):
    tree[index] = value
  else:
    tree.append(value)
  return tree


def _get_tree_def(tree: Branch) -> TreeDefBranch:
  """Creates a TreeDefBranch of analogous type to that of 'tree'."""
  # Match the type for a set of basic cases.
  if isinstance(tree, (list, tuple, dict)):
    return type(tree)()
  # Otherwise simplify to a list or dict to address immutability.
  elif isinstance(tree, Sequence):
    return list()
  elif isinstance(tree, Mapping):
    return dict()
  else:
    raise TypeError(f"Cannot convert type '{type(tree)}' into a TreeDef.")


def _get_generator(tree: Branch, sort_keys: bool):
  """Abstracts iterating over a pytree."""
  if isinstance(tree, Sequence):
    return enumerate(tree)
  else:
    if sort_keys:
      if not all([isinstance(key, str) for key in tree.keys()]):
        raise TypeError("Non-string keys are not supported when sort_keys=True")
      return sorted(tree.items(), key=lambda items: items[0])
    else:
      return tree.items()


def tree_flatten(tree: Tree,
                 is_leaf: Callable[[Tree], bool] = is_leaf,
                 sort_keys: bool = True,
                 in_place: bool = False) -> Tuple[List[Leaf], TreeDef]:
  """Flattens 'tree' into a list of leafs and a TreeDef that can reconstruct it.

  The types of branches in the pytree are not guaranteed to be preserved, but
  are guaranteed to remain subclasses of Sequence and Mapping. Namely, if
  MyCustomMapping is passed to tree_unflatten(*tree_flatten(tree)), then the
  type of the returned tree will be Dict.

  Arguments:
    tree:
      The pytree to flatten.
    is_leaf:
      A callable that determines what is interpreted as a leaf. By default,
      anything that isn't a subclass of Sequence or Mapping is interpreted as a
      leaf. Being less restrictive about what is considered a leaf is supported,
      but being more restrictive is not guaranteed to work.
    sort_keys:
      Whether or not to mirror the XLA implementation and sort all Mapping keys
      alphabetically. If true, then all keys must be strings.
    in_place:
      Whether or not to copy all data to ensure immutability.

  Raises:
    TypeError:
      if 'sort_keys' is True and there are non-string keys.
  """
  if not in_place:
    tree = copy.deepcopy(tree)

  if is_leaf(tree):
    return [tree], None

  leaves = []
  tree_def = _get_tree_def(tree)
  for index, child in _get_generator(tree, sort_keys):
    child_leaves, child_tree_def = tree_flatten(child,
                                                is_leaf,
                                                sort_keys,
                                                in_place=True)
    leaves.extend(child_leaves)
    tree_def = _set_element(tree_def, index, child_tree_def)
  return leaves, tree_def  # pytype: disable=bad-return-type


def _tree_unflatten(leaves: Sequence[Leaf], tree_def: TreeDef,
                    leaf_index: int) -> Tuple[Tree, int]:
  if tree_def is None:
    # tree_def represents a leaf.
    return leaves[leaf_index], leaf_index + 1

  for tree_index, child_tree_def in _get_generator(tree_def, sort_keys=False):
    child_tree, leaf_index = _tree_unflatten(leaves, child_tree_def, leaf_index)
    tree_def = _set_element(tree_def, tree_index, child_tree)
  return tree_def, leaf_index


def tree_unflatten(leaves: Sequence[Leaf],
                   tree_def: TreeDef,
                   in_place: bool = False) -> Tree:
  """Unflattens the outputs of 'tree_flatten'.

  Arguments:
    leaves:
      The sequence of leaves to unflatten.
    tree_def:
      A TreeDef with 'len(leaves)' TreeDefLeafs.
    in_place:
      Whether or not to copy all data to ensure immutability.

  Raises:
    ValueError:
      if 'leaves' has more values than tree_def expects.
  """
  if not in_place:
    leaves = copy.deepcopy(leaves)
    tree_def = copy.deepcopy(tree_def)

  tree, leaf_index = _tree_unflatten(leaves, tree_def, 0)
  if leaf_index != len(leaves):
    raise ValueError(
        f"'leaves' had {leaf_index - len(leaves)} more leaves than 'tree_def'")
  return tree


def tree_map(function: Callable[[Any], Any],
             tree: Tree,
             is_leaf: Callable[[Tree], bool] = is_leaf,
             sort_keys: bool = True,
             in_place: bool = False) -> Tree:
  """Applies 'function' to all the leaves in 'tree'.

  Arguments:
    function:
      The callable to apply to the leaves of 'tree'.
    tree:
      The pytree whose leaves to apply 'function' to.
    is_leaf:
      A callable that determines what is interpreted as a leaf. By default,
      anything that isn't a subclass of Sequence or Mapping is interpreted as a
      leaf. Being less restrictive about what is considered a leaf is supported,
      but being more restrictive is not guaranteed to work.
    sort_keys:
      Whether or not to mirror the XLA implementation and sort all Mapping keys
      alphabetically. If true, then all keys must be strings.
    in_place:
      Whether or not to copy all data to ensure immutability.

  Raises:
    TypeError:
      if 'sort_keys' is True and there are non-string keys.
  """
  leaves, tree_def = tree_flatten(tree, is_leaf, sort_keys, in_place)
  leaves = list(map(function, leaves))
  return tree_unflatten(leaves, tree_def)
