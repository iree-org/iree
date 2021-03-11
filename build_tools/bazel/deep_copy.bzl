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

def _is_dict(x):
    return type(x) == type({})

def _is_list(x):
    return type(x) == type([])

# Starlark doesn't support recursion, so we 'simulate' it by copying and pasting.
def _deep_copy_recursion_depth_3(x):
    """Helper method for deep_copy()."""
    if _is_dict(x) or _is_list(x):
        fail("Cannot make a deep copy of containers nested too deeply")
    return x

def _deep_copy_recursion_depth_2(x):
    """Helper method for deep_copy()."""
    if _is_dict(x):
        return {key: _deep_copy_recursion_depth_3(value) for key, value in x.items()}
    if _is_list(x):
        return [_deep_copy_recursion_depth_3(value) for value in x]
    return x

def _deep_copy_recursion_depth_1(x):
    """Helper method for deep_copy()."""
    if _is_dict(x):
        return {key: _deep_copy_recursion_depth_2(value) for key, value in x.items()}
    if _is_list(x):
        return [_deep_copy_recursion_depth_2(value) for value in x]
    return x

def deep_copy(x):
    """Returns a copy of the argument, making a deep copy if it is a container.

    Args:
      x: (object) value to copy. If it is a container with nested containers as
         elements, the maximum nesting depth is restricted to three (e.g.,
         [[[3]]] is okay, but not [[[[4]]]]). If it is a struct, it is treated
         as a value type, i.e., only a shallow copy is made.
    Returns:
      A copy of the argument.
    """
    if _is_dict(x):
        return {key: _deep_copy_recursion_depth_1(value) for key, value in x.items()}
    if _is_list(x):
        return [_deep_copy_recursion_depth_1(value) for value in x]
    return x
