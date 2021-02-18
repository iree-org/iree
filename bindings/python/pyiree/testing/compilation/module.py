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
from typing import Dict, Set

from .stages import Stage, Representation

__all__ = [
    "CompilationDefModule",
]


# TODO(meadowlark): Add decorators that set exported_names and expected failures
class CompilationDefModule(object):
  """Base class for modules defining exported names and expected failures."""
  representation: Representation = None
  exported_names: Set[str] = set()
  expected_compilation_failures: Dict[str, Stage] = dict()

  @classmethod
  def get_path(cls, test_dir: str) -> str:
    return os.path.join(test_dir,
                        f"{cls.__name__}.{cls.representation.file_extension}")

  @classmethod
  def save(cls, test_dir: str):
    raise NotImplementedError()
