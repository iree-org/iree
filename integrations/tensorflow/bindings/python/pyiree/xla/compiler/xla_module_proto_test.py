# Copyright 2020 Google LLC
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
import numpy as np
from pyiree.xla import compiler

# pylint: disable=g-direct-tensorflow-import
import tensorflow.compiler.xla.python.xla_client as xla_client
# pylint: enable=g-direct-tensorflow-import

ops = xla_client.ops


class RuntimeTest(absltest.TestCase):

  def testXLA(self):
    """Tests that a basic saved model to XLA workflow grossly functions.

    This is largely here to verify that everything is linked in that needs to be
    and that there are not no-ops, etc.
    """
    # Generate a sample XLA computation.
    builder = xla_client.XlaBuilder("testbuilder")
    in_shape = np.array([4], dtype=np.float32)
    in_feed = ops.Parameter(builder, 0, xla_client.shape_from_pyval(in_shape))
    result = ops.Add(in_feed, ops.Constant(builder, np.float32(1.0)))
    xla_computation = builder.Build(result)

    # Load into XLA Module.
    module = compiler.xla_load_module_proto(xla_computation)

    # Validate imported ASM.
    xla_asm = module.to_asm()
    print("XLA ASM: ", xla_asm)
    self.assertRegex(xla_asm, "mhlo.add")


if __name__ == "__main__":
  absltest.main()
