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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import pyiree as iree
import pyiree.compiler2
from ..utils import ParsableConstants

__all__ = [
    "Representation",
    "Representations",
    "Stage",
    "Stages",
    "save_split_methods",
]


class Representation:

  def __init__(self, lower_name: str, upper_name: str, extension: str):
    self.lower_name = lower_name
    self.upper_name = upper_name
    self.extension = extension
    self.file_extension = f"{lower_name}.{extension}"
    self.methods_extension = f"{lower_name}.methods"

  def __str__(self):
    return self.lower_name

  def __repr__(self):
    return (f"Representation({repr(self.lower_name)}, {repr(self.upper_name)}, "
            f"{repr(self.extension)})")


class Representations(ParsableConstants):
  """Representations for the core IREE compiler."""
  MHLO = Representation("mhlo", "MHLO", "mlir")
  VMLA = Representation("vmla", "VMLA", "vmfb")
  LLVMAOT = Representation("llvmaot", "LLVMAOT", "vmfb")
  VULKAN = Representation("vulkan", "Vulkan", "vmfb")


Pipeline = Callable[[str, str, Sequence[str]], None]


def _mhlo_to_vmla(source_path: str, target_path: str,
                  exported_names: Sequence[str]) -> None:
  iree.compiler2.compile_file(source_path,
                              output_file=target_path,
                              target_backends=["vmla"])


def _mhlo_to_llvmaot(source_path: str, target_path: str,
                     exported_names: Sequence[str]) -> None:
  iree.compiler2.compile_file(source_path,
                              output_file=target_path,
                              target_backends=["dylib-llvm-aot"])


def _mhlo_to_vulkan(source_path: str, target_path: str,
                    exported_names: Sequence[str]) -> None:
  iree.compiler2.compile_file(source_path,
                              output_file=target_path,
                              target_backends=["vulkan-*"])


def save_split_methods(name_to_artifact: Dict[str, Any], target_path: str,
                       extension: str):
  """Handles saving for stages that separately compile multiple methods."""
  if target_path.endswith(".methods"):
    os.makedirs(target_path, exist_ok=True)
    for method, artifact in name_to_artifact.items():
      with open(os.path.join(target_path, f"{method}.{extension}"), "wb") as f:
        f.write(artifact)
  else:
    if len(name_to_artifact) > 1:
      raise ValueError("Lowering multiple split methods into a single file is "
                       "not supported.")
    artifact = list(name_to_artifact.values())[0]
    with open(os.path.join(target_path), "wb") as f:
      f.write(artifact)


class Stage:
  """Represents a translation from one Representation to another."""

  def __init__(self,
               source: Representation,
               target: Representation,
               pipeline: Pipeline,
               splits_methods: bool = False):
    self.source = source
    self.target = target
    self._pipeline = pipeline
    self.lower_name = f"{source.lower_name}_to_{target.lower_name}"
    self.upper_name = f"{source.upper_name}To{target.upper_name}"
    self.splits_methods = splits_methods

  def pipeline(self, source_path: str, target_path: str,
               exported_names: Sequence[str]):
    """Translate source_path to the target representation."""
    if source_path.endswith(self.source.methods_extension):
      if not target_path.endswith(self.target.methods_extension):
        raise ValueError("Cannot lower a methods directory into a single file.")
      os.makedirs(target_path, exist_ok=True)

      for name in exported_names:
        self._pipeline(
            os.path.join(source_path, f"{name}.{self.source.file_extension}"),
            os.path.join(target_path, f"{name}.{self.target.file_extension}"),
            exported_names=[name],
        )
    else:
      os.makedirs(os.path.dirname(target_path), exist_ok=True)
      self._pipeline(source_path, target_path, exported_names)

  def _validate_source_path(self, source_path: str):
    basename = os.path.basename(source_path)
    if not (basename.endswith(self.source.file_extension) or
            basename.endswith(self.source.methods_extension)):
      raise ValueError(
          f"Did not recognize source_path's file extension: '{source_path}'. "
          f"Expected one of '{self.source.file_extension}' or "
          f"'{self.source.methods_extension}'.")

  def get_target_extension(self, source_path: str) -> str:
    self._validate_source_path(source_path)
    basename = os.path.basename(source_path)
    if self.splits_methods or basename.endswith(self.source.methods_extension):
      return self.target.methods_extension
    else:
      return self.target.file_extension

  def get_target_name(self, source_path: str) -> str:
    """Returns the name for the path to the target IR."""
    self._validate_source_path(source_path)
    basename = os.path.basename(source_path)
    if basename.endswith(self.source.file_extension):
      return basename[:-len(f".{self.source.file_extension}")]
    else:
      return basename[:-len(f".{self.source.methods_extension}")]

  def get_target_dir(self,
                     source_path: str,
                     nest_under_partial_lowerings: bool = False) -> str:
    """Gets the directory to store the compiled target artifacts under."""
    target_dir = os.path.dirname(source_path)
    # Optionally ensure that the target is nested under a separate dir.
    if nest_under_partial_lowerings and not "__partial_lowerings" in target_dir:
      target_dir = os.path.join(target_dir, "__partial_lowerings")
    return target_dir

  def get_target_path(self, source_path: str) -> str:
    target_dir = self.get_target_dir(source_path)
    target_name = self.get_target_name(source_path)
    target_extension = self.get_target_extension(source_path)
    return os.path.join(target_dir, f"{target_name}.{target_extension}")

  def __str__(self):
    return f"Stage({self.lower_name})"

  def __repr__(self):
    return (f"Stage({repr(self.source)}, {repr(self.target)}, "
            f"{self.pipeline.__name__})")


class Stages(ParsableConstants):
  """Compilation stages for the core IREE compiler."""
  MHLO_TO_VMLA = Stage(Representations.MHLO,
                       Representations.VMLA,
                       pipeline=_mhlo_to_vmla)
  MHLO_TO_LLVMAOT = Stage(Representations.MHLO,
                          Representations.LLVMAOT,
                          pipeline=_mhlo_to_llvmaot)
  MHLO_TO_VULKAN = Stage(Representations.MHLO,
                         Representations.VULKAN,
                         pipeline=_mhlo_to_vulkan)
