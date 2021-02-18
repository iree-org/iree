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

import pyiree as iree
import pyiree.testing
from .compilation.stages import Stages

__all__ = [
    "Lowerings",
]


# TODO(meadowlark): Add runtime_module in execution testing PR.
class Lowerings(iree.testing.Lowerings):
  REFERENCE = iree.testing.Lowering([])
  VMLA_VIA_MHLO = iree.testing.Lowering([
      Stages.JAX_TO_MHLO,
      Stages.MHLO_TO_VMLA,
  ])
  LLVMAOT_VIA_MHLO = iree.testing.Lowering([
      Stages.JAX_TO_MHLO,
      Stages.MHLO_TO_LLVMAOT,
  ])
  VULKAN_VIA_MHLO = iree.testing.Lowering([
      Stages.JAX_TO_MHLO,
      Stages.MHLO_TO_VULKAN,
  ])
