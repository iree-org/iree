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

from pyiree import binding as binding


class CompilerTest(absltest.TestCase):

  def testParseError(self):
    ctx = binding.compiler.CompilerContext()
    with self.assertRaises(ValueError):
      ctx.parse_asm("""FOOBAR: I SHOULD NOT PARSE""")
    diag_str = ctx.get_diagnostics()
    self.assertRegex(diag_str, "custom op 'FOOBAR' is unknown")

  def testParseAndCompileToSequencer(self):
    ctx = binding.compiler.CompilerContext()
    input_module = ctx.parse_asm("""
      func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
            attributes { iree.module.export } {
          %0 = "xla_hlo.mul"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          return %0 : tensor<4xf32>
      }
      """)
    binary = input_module.compile_to_sequencer_blob()
    self.assertTrue(binary)


if __name__ == '__main__':
  absltest.main()
