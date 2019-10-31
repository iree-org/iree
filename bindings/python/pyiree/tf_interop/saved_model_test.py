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

import importlib
import os
import sys
import tempfile

import pyiree

# Determine if compiled with tf_interop support.
if not hasattr(pyiree, "tf_interop"):
  print("Not running tests because tf_interop support not compiled")
  sys.exit(0)

# Dynamically import tensorflow.
try:
  # Use a dynamic import so as to avoid hermetic dependency analysis
  # (i.e. we only want the tensorflow from the environment).
  tf = importlib.import_module("tensorflow")
  # Just in case if linked against a pre-V2 defaulted version.
  tf.enable_v2_behavior()
  tf = tf.compat.v2
except ImportError:
  print("Not running tests because tensorflow is not available")
  sys.exit(0)


class StatefulModule(tf.Module):

  def __init__(self):
    self.v = tf.Variable([4], dtype=tf.float32)

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def add(self, a, b):
    return tf.tanh(self.v * a + b)


class RuntimeTest(tf.test.TestCase):

  def testLoadSavedModelToXlaPipeline(self):
    """Tests that a basic saved model to XLA workflow grossly functions.

    This is largely here to verify that everything is linked in that needs to be
    and that there are not no-ops, etc.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
      sm_dir = os.path.join(temp_dir, "simple.sm")
      print("Saving to:", sm_dir)
      my_module = StatefulModule()
      options = tf.saved_model.SaveOptions(save_debug_info=True)
      tf.saved_model.save(my_module, sm_dir, options=options)

      # Load it up.
      ctx = pyiree.CompilerContext()
      input_module = pyiree.tf_load_saved_model(ctx, sm_dir)
      input_asm = input_module.to_asm()
      print("LOADED ASM:\n", input_asm)
      # Should have out exported name and have executor islands.
      self.assertRegex(input_asm,
                       r"""tf_saved_model.exported_names = \["add"\]""")
      self.assertRegex(input_asm, r"""tf_executor\.island""")

      # Run the necessary lowering passes. Makes sure that these are linked in.
      input_module.run_pass_pipeline([
          "tf-executor-graph-pruning",
          "tf-standard-pipeline",
          "canonicalize",
      ])
      lowered_asm = input_module.to_asm()
      print("LOWERED ASM:\n", lowered_asm)
      # Should have collapsed all executor islands.
      self.assertNotRegex(lowered_asm, r"""tf_executor\.island""")

      # And legalize to XLA.
      input_module.run_pass_pipeline([
          "xla-legalize-tf",
      ])
      xla_asm = input_module.to_asm()
      print("XLA ASM:", xla_asm)
      self.assertRegex(xla_asm, "xla_hlo.tanh")


if __name__ == "__main__":
  tf.test.main()
