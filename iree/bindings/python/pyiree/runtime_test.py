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

from pyiree import binding as binding


class RuntimeTest(absltest.TestCase):

  def testModuleAndFunction(self):

    blob = binding.compiler.compile_module_from_asm("""
    func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
          attributes { iree.module.export } {
        %0 = "xla_hlo.mul"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
    }
    """)
    self.assertTrue(blob)
    print("Module blob:", blob)
    m = binding.vm.create_module_from_blob(blob)
    print("Module:", m)
    print("Module name:", m.name)
    self.assertEqual("module", m.name)

    # Function 0.
    f = m.lookup_function_by_ordinal(0)
    print("Function 0:", f)
    self.assertEqual("simple_mul", f.name)
    sig = f.signature
    self.assertEqual(2, sig.argument_count)
    self.assertEqual(1, sig.result_count)

    # Function 1.
    f = m.lookup_function_by_ordinal(1)
    self.assertIs(f, None)

    # By name.
    f = m.lookup_function_by_name("simple_mul")
    self.assertEqual("simple_mul", f.name)
    sig = f.signature
    self.assertEqual(2, sig.argument_count)
    self.assertEqual(1, sig.result_count)

    # By name not found.
    f = m.lookup_function_by_name("not_here")
    self.assertIs(f, None)


if __name__ == "__main__":
  absltest.main()
