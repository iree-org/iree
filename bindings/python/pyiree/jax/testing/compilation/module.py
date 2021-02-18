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

import os
import pickle
from typing import Any, Dict, Sequence

import numpy as np

import pyiree as iree
import pyiree.testing.compilation
from .stages import Representations

__all__ = [
    "ArraySpec",
    "CompilationDefModule",
]


class ArraySpec(object):

  def __init__(self, shape: Sequence[int], dtype: Any = np.float32):
    self.shape = shape
    self.dtype = dtype


class CompilationDefModule(iree.testing.compilation.CompilationDefModule):
  """Module for defining JAX compilation info."""
  representation: iree.testing.compilation.Representation = Representations.JAX
  exported_name_to_input_args: Dict[str, Sequence[Any]] = dict()
  exported_name_to_input_signature: Dict[str, ArraySpec] = dict()

  @classmethod
  def save(cls, test_dir: str):
    os.makedirs(test_dir, exist_ok=True)
    with open(cls.get_path(test_dir), "wb") as f:
      pickle.dump(cls(), f)
