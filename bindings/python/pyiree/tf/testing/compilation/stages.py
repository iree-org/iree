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
from typing import Sequence

from absl import logging
import tensorflow as tf

import pyiree as iree
import pyiree.compiler2.tf
import pyiree.testing.compilation

__all__ = [
    "Stages",
]


class Representations(iree.testing.compilation.Representations):
  """Extends IREE's core representations with those for TensorFlow."""
  TF = iree.testing.compilation.Representation("tf", "TF", "saved_model")
  TFLITE = iree.testing.compilation.Representation("tflite", "TFLite",
                                                   "flatbuffer")


def _tf_to_tflite(source_path: str, target_path: str,
                  exported_names: Sequence[str]) -> None:
  module = tf.saved_model.load(source_path)
  name_to_flatbuffer = dict()
  failed_names = []

  for name in exported_names:
    try:
      logging.info("Attempting to convert '%s' to tflite...", name)
      function = getattr(module, name).get_concrete_function()
      converter = tf.lite.TFLiteConverter.from_concrete_functions([function])
      name_to_flatbuffer[name] = converter.convert()
      logging.info("...converted '%s' to tflite.", name)
    except Exception as e:
      logging.error("Failed to convert '%s' to tflite.", name)
      logging.error("TFLite excpetion: %s", e)
      failed_names.append(name)

  iree.testing.compilation.save_split_methods(
      name_to_flatbuffer, target_path, Representations.TFLITE.file_extension)

  if failed_names:
    raise RuntimeError(
        f"Failed to convert the following methods to tflite: {name}")

  return target_path


def _tf_to_mhlo(source_path: str, target_path: str,
                exported_names: Sequence[str]) -> None:
  iree.compiler2.tf.compile_saved_model(source_path,
                                        output_file=target_path,
                                        exported_names=exported_names,
                                        import_only=True)


class Stages(iree.testing.compilation.Stages):
  """Compilation stages for the TensorFlow frontend."""
  TF_TO_TFLITE = iree.testing.compilation.Stage(Representations.TF,
                                                Representations.TFLITE,
                                                pipeline=_tf_to_tflite,
                                                splits_methods=True)
  TF_TO_MHLO = iree.testing.compilation.Stage(Representations.TF,
                                              Representations.MHLO,
                                              pipeline=_tf_to_mhlo)
