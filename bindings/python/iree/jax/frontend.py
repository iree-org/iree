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

import functools

import iree.compiler.xla
import iree.rt

try:
  import jax
except ModuleNotFoundError as e:
  raise ModuleNotFoundError("iree.jax requires 'jax' and 'jaxlib' to be "
                            "installed in your python environment.") from e

# pytype thinks iree.jax is jax.
# pytype: disable=module-attr

__all__ = [
    "aot",
    "is_available",
    "jit",
]

_BACKENDS = ["vmla", "llvmaot", "vulkan"]
_BACKEND_TO_DRIVER = {
    "vmla": "vmla",
    "llvmaot": "dylib",
    "vulkan": "vulkan",
}
_BACKEND_TO_TARGETS = {
    "vmla": ("vmla",),
    "llvmaot": ("dylib-llvm-aot",),
    "vulkan": ("vulkan-*",),
}


def is_available():
  """Determine if the IREE–XLA compiler are available for JAX."""
  return iree.compiler.xla.is_available()


def aot(function, *args, **options):
  """Traces and compiles a function, flattening the input args.

  This is intended to be a lower-level interface for compiling a JAX function to
  IREE without setting up the runtime bindings to use it within Python. A common
  usecase for this is compiling to Android (and similar targets).

  Args:
    function: The function to compile.
    args: The inputs to trace and compile the function for.
    **kwargs: Keyword args corresponding to xla.ImportOptions or CompilerOptions
  """
  xla_comp = jax.xla_computation(function)(*args)
  hlo_proto = xla_comp.as_serialized_hlo_module_proto()
  binary = iree.compiler.xla.compile_str(hlo_proto, **options)
  return binary


# A more JAX-native approach to jitting would be desireable here, however
# implementing that reasonably would require using JAX internals, particularly
# jax.linear_util.WrappedFun and helpers. The following is sufficient for many
# usecases for the time being.


class _JittedFunction:

  def __init__(self, function, driver: str, **options):
    self._function = function
    self._driver_config = iree.rt.Config(driver)
    self._options = options
    self._memoized_signatures = {}

  def _get_signature(self, args_flat, in_tree):
    args_flat = [iree.rt.normalize_value(arg) for arg in args_flat]
    return tuple((arg.shape, arg.dtype) for arg in args_flat) + (in_tree,)

  def _wrap_and_compile(self, signature, args_flat, in_tree):
    """Compiles the function for the given signature."""

    def wrapped_function(*args_flat):
      args, kwargs = jax.tree_unflatten(in_tree, args_flat)
      return self._function(*args, **kwargs)

    # Compile the wrapped_function to IREE.
    binary = aot(wrapped_function, *args_flat, **self._options)
    cpp_vm_module = iree.rt.VmModule.from_flatbuffer(binary)
    module = iree.rt.load_module(cpp_vm_module, config=self._driver_config)

    # Get the output tree so it can be reconstructed from the outputs of the
    # compiled module. Duplicating execution here isn't ideal, and could
    # probably be avoided using internal APIs.
    args, kwargs = jax.tree_unflatten(in_tree, args_flat)
    _, out_tree = jax.tree_flatten(self._function(*args, **kwargs))

    self._memoized_signatures[signature] = (binary, module, out_tree)

  def _get_compiled_artifacts(self, args, kwargs):
    """Returns the binary, loaded rt module and out_tree."""
    args_flat, in_tree = jax.tree_flatten((args, kwargs))
    signature = self._get_signature(args_flat, in_tree)

    if signature not in self._memoized_signatures:
      self._wrap_and_compile(signature, args_flat, in_tree)
    return self._memoized_signatures[signature]

  def __call__(self, *args, **kwargs):
    """Executes the function on the provided inputs, compiling if necessary."""
    args_flat, _ = jax.tree_flatten((args, kwargs))
    # Use the uncompiled function if the inputs are being traced.
    if any(issubclass(type(arg), jax.core.Tracer) for arg in args_flat):
      return self._function(*args, **kwargs)

    _, module, out_tree = self._get_compiled_artifacts(args, kwargs)
    results = module.main(*args_flat)
    if not isinstance(results, tuple):
      results = (results,)
    return jax.tree_unflatten(out_tree, results)

  def get_binary(self, *args, **kwargs):
    """Gets the IREE-compiled binary for the given inputs."""
    binary, _, _ = self._get_compiled_artifacts(args, kwargs)
    return binary


def jit(function=None, *, backend: str = "llvmaot", **options):
  """Compiles a function to the specified IREE backend."""
  if function is None:
    # 'function' will be None if @jit() is called with parens (e.g. to specify a
    # backend or **options). We return a partial function capturing these
    # options, which python will then apply as a decorator, and execution will
    # continue below.
    return functools.partial(jit, backend=backend, **options)

  # Parse the backend to more concrete compiler and runtime settings.
  if backend not in _BACKENDS:
    raise ValueError(
        f"Expected backend to be one of {_BACKENDS}, but got '{backend}'")
  driver = _BACKEND_TO_DRIVER[backend]
  if "target_backends" not in options:
    options["target_backends"] = _BACKEND_TO_TARGETS[backend]

  return functools.wraps(function)(_JittedFunction(function, driver, **options))
