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

import iree.rt
import numpy as np


class HalTest(absltest.TestCase):

  def testEnums(self):
    logging.info("MemoryType: %s", iree.rt.MemoryType)
    logging.info("HOST_VISIBLE: %s", int(iree.rt.MemoryType.HOST_VISIBLE))


if __name__ == "__main__":
  absltest.main()
