# Lint as: python3
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

from absl.testing import absltest
from pyiree import compiler

SIMPLE_MUL_ASM = """
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
      attributes { iree.module.export } {
    %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
"""


class CompilerTest(absltest.TestCase):

  def testParseError(self):
    ctx = compiler.Context()
    with self.assertRaisesRegex(ValueError, "custom op 'FOOBAR' is unknown"):
      ctx.parse_asm("""FOOBAR: I SHOULD NOT PARSE""")

  def testParseAndCompileToFlatbuffer(self):
    ctx = compiler.Context()
    input_module = ctx.parse_asm(SIMPLE_MUL_ASM)
    binary = input_module.compile()
    b = binary.bytes
    print("Flatbuffer size =", len(b))
    self.assertTrue(binary.bytes)

  def testParseAndCompileToFlatbufferText(self):
    ctx = compiler.Context()
    input_module = ctx.parse_asm(SIMPLE_MUL_ASM)
    options = compiler.CompileOptions()
    options.output_format = compiler.OutputFormat.FLATBUFFER_TEXT
    blob = input_module.compile(options=options)
    text = blob.text
    self.assertTrue(text)

  def testParseAndCompileToMlirText(self):
    ctx = compiler.Context()
    input_module = ctx.parse_asm(SIMPLE_MUL_ASM)
    options = compiler.CompileOptions()
    options.output_format = compiler.OutputFormat.MLIR_TEXT
    blob = input_module.compile(options=options)
    text = blob.text
    self.assertTrue(text)


if __name__ == "__main__":
  absltest.main()
