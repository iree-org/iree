import tempfile
from typing import Sequence

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

import pyiree as iree
import pyiree.jax.testing
from pyiree.jax.testing import ArraySpec, Stages

FLAGS = flags.FLAGS


class AQTMatmulModule(iree.jax.testing.CompilationDefModule):
  exported_names = {"aqt_matmul"}
  expected_compilation_failures = {
      "aqt_matmul": [
          Stages.MHLO_TO_LLVMAOT,
          Stages.MHLO_TO_VULKAN,
      ],
  }
  exported_name_to_input_signature = {
      "aqt_matmul": [
          ArraySpec([5, 6], np.float32),
          ArraySpec([6, 3], np.float32),
          ArraySpec([], np.float32),
      ],
  }

  def aqt_matmul(self, activation, weight, activation_scale):
    precision = 8
    lower_bound = -2**(precision - 1) + 1
    upper_bound = 2**(precision - 1) - 1

    activation_scaled = activation * activation_scale
    activation_rounded = jnp.floor(activation_scaled + jnp.array(0.5))
    activation_clipped = jnp.clip(activation_rounded, lower_bound, upper_bound)
    activation_as_int = activation_clipped.astype(jnp.int8)

    weight_scale = upper_bound / jnp.max(jnp.abs(weight))
    weight_scaled = weight * weight_scale
    weight_rounded = jnp.floor(weight_scaled + jnp.array(0.5))
    weight_as_int = weight_rounded.astype(jnp.int8)

    scaled_result = jax.lax.dot(activation_as_int,
                                weight_as_int,
                                preferred_element_type=jnp.int32)
    return scaled_result / (activation_scale * weight_scale)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # TODO(meadowlark): Move flag parsing into iree.testing after deciding on a
  # high level API.
  lowering = iree.jax.testing.Lowerings.parse(FLAGS.lowering)
  with tempfile.TemporaryDirectory() as test_dir:
    test_results, lowered_path = iree.jax.testing.run_compilation_tests(
        AQTMatmulModule, lowering, test_dir, fail_on_unexpected_success=False)


if __name__ == "__main__":
  app.run(main)
