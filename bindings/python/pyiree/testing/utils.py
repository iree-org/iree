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

from typing import Any

import pyiree as iree
import pyiree.rt

__all__ = [
    "ParsableConstants",
]


class ParsableConstants:
  """Similar to Enum but extendable and without the '.value' indirection."""

  @classmethod
  def get_members(cls):
    return set(dir(cls)) - set(dir(ParsableConstants))

  @classmethod
  def parse(cls, name: str) -> Any:
    """Returns one of the constants on the class if 'name' is a match."""
    members = cls.get_members()
    parsed_name = name.upper().replace("-", "_")
    if parsed_name not in members:
      raise ValueError(
          f"Expected one of {members} (case insensitive), but got {name}.")
    return getattr(cls, parsed_name)
