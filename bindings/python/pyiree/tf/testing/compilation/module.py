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

import tensorflow as tf

import pyiree as iree
import pyiree.testing.compilation
from .stages import Representations

__all__ = [
    "CompilationDefModule",
]


class CompilationDefModule(iree.testing.compilation.CompilationDefModule,
                           tf.Module):
  """TensorFlow module for defining exported names and expected failures."""
  representation: iree.testing.compilation.Representation = Representations.TF

  @classmethod
  def save(cls, test_dir: str):
    instance = cls()
    path = cls.get_path(test_dir)
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(instance, path, options=options)
