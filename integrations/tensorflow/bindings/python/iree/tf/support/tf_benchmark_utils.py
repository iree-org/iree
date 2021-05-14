# Lint as: python3
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
"""Benchmark utilities interop with TensorFlow."""

# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=unsupported-assignment-operation

# TODO(#4131) python>=3.7: Use postponed type annotations.

import collections
import os
import tempfile
from typing import Sequence, Set, Type

from absl import flags
from absl import logging
from iree.tf.support import module_utils
import tensorflow.compat.v2 as tf

flags.DEFINE_string("target_backend", None,
                    "The target backend to benchmark against.")
flags.DEFINE_list("configuration_names", None,
                  "A comma-separated list of configuration names")
flags.DEFINE_list(
    "compilation_flags", None,
    "A comma-separated list of semicolon-separated compilation flags for "
    "each configuration")
flags.DEFINE_list(
    "runtime_flags", None,
    "A comma-separated list of semicolon-separated runtime flags for "
    "each configuration")
flags.DEFINE_string(
    "artifacts_dir", None,
    "Specifies a directory to dump compilation artifacts and traces to. "
    "Defaults to the OS's tempdir.")

FLAGS = flags.FLAGS


def _setup_artifacts_dir(relative_artifacts_dir: str) -> str:
  parent_dirs = [
      FLAGS.artifacts_dir,
      os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"),
      os.environ.get("TEST_TMPDIR"),
      os.path.join(tempfile.gettempdir(), "iree", "benchmarks"),
  ]
  # Use the most preferred path in parent_dirs that isn't None.
  parent_dir = next(parent for parent in parent_dirs if parent is not None)

  artifacts_dir = os.path.join(parent_dir, relative_artifacts_dir)
  logging.info("Saving compilation artifacts and traces to '%s'", artifacts_dir)
  os.makedirs(artifacts_dir, exist_ok=True)
  return artifacts_dir


def get_target_backend_configs() -> Sequence[module_utils.BackendInfo]:
  """Gets the BackendInfo instances for each target backend configuration.

  Returns:
    Sequence of BackendInfo that should be used.
  """
  if FLAGS.target_backend is None:
    raise ValueError("--target_backend must be specified")

  if FLAGS.configuration_names is None:
    raise ValueError("--configuration_names must be specified")
  configuration_names = FLAGS.configuration_names

  compilation_flags = FLAGS.compilation_flags
  if compilation_flags is None:
    compilation_flags = [""] * len(configuration_names)

  runtime_flags = FLAGS.runtime_flags
  if runtime_flags is None:
    runtime_flags = [""] * len(configuration_names)

  if len(configuration_names) != len(compilation_flags) or len(
      configuration_names) != len(runtime_flags):
    raise ValueError(
        "--configuration_names, --compilation_flags, and --runtime_flags "
        "must have the same number of elements")

  configs = []
  for name, cflags, rflags in zip(configuration_names, compilation_flags,
                                  runtime_flags):
    cflags = cflags.split(";")
    rflags = rflags.split(";")

    logging.info(f"configuration: {name}")
    logging.info(f"compilation flags: {cflags}")
    logging.info(f"runtime flags: {rflags}")

    backend_id = f"{FLAGS.target_backend}__{name}"
    configs.append(
        module_utils.BackendInfo(FLAGS.target_backend, backend_id, cflags,
                                 rflags))

  return configs


def compose_and_write_flagfile(
    path: str,
    driver: str,
    entry_function: str,
    function_inputs: Sequence[str],
    additional_args: Sequence[str] = ()) -> None:
  flagfile = [
      f"--module_file=compiled.vmfb",
      f"--driver={driver}",
      f"--entry_function={entry_function}",
  ]
  flagfile.extend([f"--function_input={input}" for input in function_inputs])
  flagfile.extend(additional_args)

  with open(os.path.join(path, "flagfile"), "w") as f:
    f.writelines(line + "\n" for line in flagfile)


def get_mlir_tensor_type(shape: Sequence[int], dtype: tf.dtypes.DType):

  def convert_dtype(dtype: tf.dtypes.DType):
    if dtype.name.startswith("float"):
      return "f" + dtype.name[5:]
    if dtype.name.startswith("int"):
      return "i" + dtype.name[3:]
    raise ValueError(f"Unimplemented conversion for dtype: {dtype}")

  return "x".join([str(d) for d in shape]) + "x" + convert_dtype(dtype)


# TODO(#4131) python>=3.7: Consider using a (frozen) dataclass.
Modules = collections.namedtuple("Modules", ["target_modules", "artifacts_dir"])


def compile_tf_module(module_class: Type[tf.Module],
                      exported_name: str,
                      input_shapes_dtypes: Sequence[tuple[Sequence[int],
                                                          tf.dtypes.DType]],
                      relative_artifacts_dir: str = None) -> Modules:
  """Compiles module_class to each backend that we test.

  Args:
    module_class: the tf.Module subclass to compile.
    exported_name: a string representing which of module_class's functions to
      compile.
    input_shapes_dtypes: a sequence of input tensors' shapes and dtypes.
    relative_artifacts_dir: optional string specifying where to save compilation
      artifacts within the artifacts_dir. If it is not specified then
      module_class.__name__ will be used.

  Returns:
    A 'Modules' namedtuple containing the target modules for each configuration
    and artifacts directory.
  """
  # Setup the directory for saving compilation artifacts and traces.
  if relative_artifacts_dir is None:
    relative_artifacts_dir = module_class.__name__
  artifacts_dir = _setup_artifacts_dir(relative_artifacts_dir)

  target_backend_configs = get_target_backend_configs()

  compile_backend = lambda backend_info: backend_info.compile_from_class(
      module_class, [exported_name], artifacts_dir, create_runtime=False)

  target_modules = []
  for config in target_backend_configs:
    target_modules.append(compile_backend(config))
    flagfile_path = os.path.join(artifacts_dir, config.backend_id)
    function_inputs = [
        get_mlir_tensor_type(shape, dtype)
        for shape, dtype in input_shapes_dtypes
    ]
    compose_and_write_flagfile(flagfile_path, config.driver, exported_name,
                               function_inputs, config.runtime_flags)

  return Modules(target_modules, artifacts_dir)


def compile_tf_signature_def_saved_model(
    saved_model_dir: str, saved_model_tags: Set[str], module_name: str,
    exported_name: str, input_names: Sequence[str],
    input_shapes_dtypes: Sequence[tuple[Sequence[int], tf.dtypes.DType]],
    output_names: Sequence[str]) -> Modules:
  """Compiles a SignatureDef SavedModel to each backend that we test.

  Args:
    saved_model_dir: Directory of the saved model.
    saved_model_tags: Optional set of tags to use when loading the model.
    module_name: A name for this compiled module.
    backend_info: BackendInfo with the details for compiling the saved model.
    exported_name: A str representing the signature on the saved model to
      compile.
    input_names: A sequence of kwargs to feed to the saved model.
    input_shapes_dtypes: a sequence of input tensors' shapes and dtypes.
    output_names: A sequence of named outputs to extract from the saved model.

  Returns:
    A 'Modules' namedtuple containing the reference module, target modules and
    artifacts directory.
  """
  # Setup the directory for saving compilation artifacts and traces.
  artifacts_dir = _setup_artifacts_dir(module_name)

  target_backend_configs = get_target_backend_configs()

  compile_backend = (lambda backend_info: backend_info.
                     compile_signature_def_saved_model(saved_model_dir,
                                                       saved_model_tags,
                                                       module_name,
                                                       exported_name,
                                                       input_names,
                                                       output_names,
                                                       artifacts_dir,
                                                       create_runtime=False))

  target_modules = []
  for config in target_backend_configs:
    target_modules.append(compile_backend(config))
    flagfile_path = os.path.join(artifacts_dir, config.backend_id)
    function_inputs = [
        get_mlir_tensor_type(shape, dtype)
        for shape, dtype in input_shapes_dtypes
    ]
    compose_and_write_flagfile(flagfile_path, config.driver, exported_name,
                               function_inputs, config.runtime_flags)

  return Modules(target_modules, artifacts_dir)
