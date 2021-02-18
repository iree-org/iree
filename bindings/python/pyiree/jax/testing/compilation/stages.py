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
from typing import Sequence

import jax
import jax.numpy as jnp

import pyiree as iree
import pyiree.compiler2.xla
import pyiree.testing.compilation

__all__ = [
    "Stages",
]


class Representations(iree.testing.compilation.Representations):
  """Extends IREE's core representations with those for JAX."""
  JAX = iree.testing.compilation.Representation("jax", "JAX", "pkl")


def _jax_to_mhlo(source_path: str, target_path: str, exported_names: str):
  with open(source_path, "rb") as f:
    module = pickle.load(f)

  names_to_mlir = dict()
  for name in exported_names:
    # Get input_args for traced compilation.
    if name in module.exported_name_to_input_args:
      input_args = module.exported_name_to_input_args[name]
    else:
      input_args = []
      for array_spec in module.exported_name_to_input_signature[name]:
        input_args.append(jnp.zeros(array_spec.shape, array_spec.dtype))
    names_to_mlir[name] = iree.jax.aot(getattr(module, name),
                                       *input_args,
                                       import_only=True)

  iree.testing.compilation.save_split_methods(
      names_to_mlir, target_path, Representations.MHLO.file_extension)


class Stages(iree.testing.compilation.Stages):
  """Compilation stages for the JAX frontend."""
  JAX_TO_MHLO = iree.testing.compilation.Stage(Representations.JAX,
                                               Representations.MHLO,
                                               pipeline=_jax_to_mhlo,
                                               splits_methods=True)
