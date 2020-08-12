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

from absl import logging
from absl.testing import absltest

import numpy as np
from pyiree import rt


class HalTest(absltest.TestCase):

  def testEnums(self):
    logging.info("MemoryType: %s", rt.MemoryType)
    logging.info("HOST_VISIBLE: %s", int(rt.MemoryType.HOST_VISIBLE))

  def testAllocateHeap(self):
    b = rt.HalBuffer.allocate_heap(
        memory_type=int(rt.MemoryType.HOST_LOCAL),
        usage=int(rt.BufferUsage.ALL),
        allocation_size=4096)
    self.assertIsNot(b, None)
    b.fill_zero(0, 4096)
    shape = rt.Shape([1, 1024])
    unused_bv = b.create_view(shape, 4)

  def testStrideCalculation(self):
    b = rt.HalBuffer.allocate_heap(
        memory_type=int(rt.MemoryType.HOST_LOCAL),
        usage=int(rt.BufferUsage.ALL),
        allocation_size=4096)
    self.assertIsNot(b, None)
    b.fill_zero(0, 4096)
    shape = rt.Shape([16, 1, 8, 4, 2])
    bv = b.create_view(shape, 4)
    self.assertEqual(
        np.array(bv.map()).strides,
        (1 * 8 * 4 * 2 * 4, 8 * 4 * 2 * 4, 4 * 2 * 4, 2 * 4, 4))


if __name__ == "__main__":
  absltest.main()
