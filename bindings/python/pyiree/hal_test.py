# Copyright 2019 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import pyiree


class HalTest(absltest.TestCase):

  def testEnums(self):
    print("MemoryType =", pyiree.binding.hal.MemoryType)
    print("HOST_VISIBLE =", int(pyiree.binding.hal.MemoryType.HOST_VISIBLE))

  def testAllocateHeap(self):
    b = pyiree.binding.hal.Buffer.allocate_heap(
        memory_type=int(pyiree.binding.hal.MemoryType.HOST_LOCAL),
        usage=int(pyiree.binding.hal.BufferUsage.ALL),
        allocation_size=4096)
    self.assertIsNot(b, None)
    b.fill_zero(0, 4096)
    shape = pyiree.binding.hal.Shape([1, 1024])
    unused_bv = b.create_view(shape, 4)


if __name__ == "__main__":
  absltest.main()
