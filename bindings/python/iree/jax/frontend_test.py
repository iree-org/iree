# Copyright 2021 Google LLC
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
import iree.jax
import iree.runtime
import jax
import jax.numpy as jnp
import numpy as np

# pytype thinks iree.jax is jax.
# pytype: disable=module-attr

TOLERANCE = {"rtol": 1e-6, "atol": 1e-6}


def normal(shape):
  return np.random.normal(0, 1, shape).astype(np.float32)


class SqrtNode:

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def apply(self, z):
    return self.x * jnp.sqrt(self.y * z)

  def tree_flatten(self):
    return ((self.x, self.y), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)


class SquareNode:

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def apply(self, z):
    return self.x * (self.y * z)**2

  def tree_flatten(self):
    return ((self.x, self.y), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)


class JAXFrontendTest(absltest.TestCase):

  def test_aot_pytree(self):

    def pytree_func(params, x):
      return jnp.max(jnp.matmul(x, params["w"]) + params["b"], 0)

    trace_args = [
        {
            "w": jnp.zeros((32, 8)),
            "b": jnp.zeros((8,))
        },
        jnp.zeros((1, 32)),
    ]
    binary = iree.jax.aot(pytree_func, *trace_args, target_backends=["vmla"])

  def test_jit_get_binary(self):

    @iree.jax.jit(backend="vmla")
    def scaled_dot_product_attention(q, k, v):
      # Subscripts:
      #   q: seq_len_q, k: seq_len_kv, d: att_depth_qk, v: att_depth_v
      qk = jnp.einsum("...qd, ...kd -> ...qk", q, k)
      soft_qk = jax.nn.softmax(qk / np.sqrt(k.shape[-1]), axis=-1)
      return jnp.einsum("...qk, ...kv -> ...qv", soft_qk, v)

    np.random.seed(0)
    inputs_1 = [normal((2, 4, 5, 8)) for _ in range(3)]
    vmla_result_1 = scaled_dot_product_attention(*inputs_1)
    jax_result_1 = scaled_dot_product_attention._function(*inputs_1)
    np.testing.assert_allclose(vmla_result_1, jax_result_1, **TOLERANCE)

    inputs_2 = [normal((2, 1, 6, 8)) for _ in range(3)]
    vmla_result_2 = scaled_dot_product_attention(*inputs_2)
    jax_result_2 = scaled_dot_product_attention._function(*inputs_2)
    np.testing.assert_allclose(vmla_result_2, jax_result_2, **TOLERANCE)

    # Test that the binaries are different.
    binary_1 = scaled_dot_product_attention.get_binary(*inputs_1)
    binary_2 = scaled_dot_product_attention.get_binary(*inputs_2)
    self.assertNotEqual(binary_1, binary_2)

    # Test that the binary gives the same result.
    cpp_vm_module = iree.runtime.VmModule.from_flatbuffer(binary_1)
    module = iree.runtime.load_module(cpp_vm_module,
                                      config=iree.runtime.Config("vmla"))
    binary_result_1 = module.main(*inputs_1)
    np.testing.assert_equal(binary_result_1, vmla_result_1)

    with self.assertRaises(ValueError):
      bad_input_shape = module.main(*inputs_2)

  def test_jit_pytree_return(self):

    @iree.jax.jit
    def apply_sqrt(pytree):
      return jax.tree_map(jnp.sqrt, pytree)

    np.random.seed(0)
    input_tree = {
        "a": [
            normal((2, 3)),
            {
                "b": normal(3)
            },
        ],
        "c": (
            {
                "d": [normal(2), normal(3)]
            },
            (normal(1), normal(4)),
        )
    }

    expected = jax.tree_map(jnp.sqrt, input_tree)
    expected_arrays, expected_tree = jax.tree_flatten(expected)
    result = apply_sqrt(input_tree)
    result_arrays, result_tree = jax.tree_flatten(result)

    self.assertEqual(expected_tree, result_tree)
    for expected_array, result_array in zip(expected_arrays, result_arrays):
      np.testing.assert_allclose(expected_array, result_array, **TOLERANCE)

  def test_iree_jit_of_iree_jit(self):

    @iree.jax.jit
    def add(a, b):
      return a + b

    @iree.jax.jit
    def mul_two(a):
      return add(a, a)

    self.assertEqual(mul_two(3), 6)

  def test_jax_jit_of_iree_jit(self):

    @iree.jax.jit
    def add(a, b):
      return a + b

    @jax.jit
    def mul_two(a):
      return add(a, a)

    self.assertEqual(mul_two(3), 6)

  def test_iree_jit_of_jax_jit(self):

    @jax.jit
    def add(a, b):
      return a + b

    @iree.jax.jit
    def mul_two(a):
      return add(a, a)

    self.assertEqual(mul_two(3), 6)

  def test_iree_jit_of_empty_iree_jit(self):

    @iree.jax.jit
    def sqrt_four():
      return jnp.sqrt(4)

    @iree.jax.jit
    def add_sqrt_four(a):
      return a + sqrt_four()

    self.assertEqual(add_sqrt_four(2), 4)

  def test_jit_pytree_method(self):

    @iree.jax.jit
    def apply_node(node, z):
      return node.apply(z)

    expected_sqrt = apply_node._function(SqrtNode(2, 3), 4)
    compiled_sqrt = apply_node(SqrtNode(2, 3), 4)
    np.testing.assert_allclose(compiled_sqrt, expected_sqrt, **TOLERANCE)

    expected_square = apply_node._function(SquareNode(2, 3), 4)
    compiled_square = apply_node(SquareNode(2, 3), 4)
    np.testing.assert_allclose(compiled_square, expected_square, **TOLERANCE)


if __name__ == "__main__":
  jax.tree_util.register_pytree_node_class(SqrtNode)
  jax.tree_util.register_pytree_node_class(SquareNode)
  absltest.main()
