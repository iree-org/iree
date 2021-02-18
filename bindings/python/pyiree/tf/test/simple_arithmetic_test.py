import tempfile
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

import pyiree as iree
import pyiree.tf.testing
from pyiree.tf.testing import Stages

FLAGS = flags.FLAGS


class SimpleArithmeticModule(iree.tf.testing.CompilationDefModule):
  exported_names = {"simple_mul", "simple_matmul", "einsum_matmul"}
  expected_compilation_failures = {
      "einsum_matmul": [
          Stages.MHLO_TO_VMLA,
          Stages.MHLO_TO_LLVMAOT,
          Stages.MHLO_TO_VULKAN,
      ]
  }

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

  @tf.function(input_signature=[
      tf.TensorSpec([128, 3072], tf.float32),
      tf.TensorSpec([3072, 256], tf.float32),
  ])
  def simple_matmul(self, a, b):
    return tf.matmul(a, b)

  @tf.function(input_signature=[
      tf.TensorSpec([128, 3072], tf.float32),
      tf.TensorSpec([3072, 256], tf.float32),
  ])
  def einsum_matmul(self, a, b):
    return tf.einsum("ij, jk -> ik", a, b)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # TODO(meadowlark): Move flag parsing into iree.testing after deciding on a
  # high level API.
  lowering = iree.tf.testing.Lowerings.parse(FLAGS.lowering)
  with tempfile.TemporaryDirectory() as test_dir:
    test_results, lowered_path = iree.tf.testing.run_compilation_tests(
        SimpleArithmeticModule,
        lowering,
        test_dir,
        fail_on_unexpected_success=False)


if __name__ == "__main__":
  app.run(main)
