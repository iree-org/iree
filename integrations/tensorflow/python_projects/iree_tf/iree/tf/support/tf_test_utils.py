# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test utilities interop with TensorFlow."""

# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=unsupported-assignment-operation

# This file uses the following abbreviations:
#   ref: reference – for the reference CompiledModule
#   tar: target - for one of the target CompiledModules

from __future__ import annotations
from dataclasses import dataclass
import collections
import copy
import itertools
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, Type, Union

from absl import flags
from absl import logging
from iree.tf.support import module_utils
from iree.tf.support import tf_utils
from iree.tf.support import trace_utils
import tensorflow.compat.v2 as tf

flags.DEFINE_string(
    "reference_backend", "tf", "The backend to treat as a source of truth."
)
flags.DEFINE_list(
    "target_backends", None, "Explicit comma-delimited list of target backends."
)
flags.DEFINE_string(
    "artifacts_dir",
    None,
    "Specifies a directory to dump compilation artifacts and traces to. "
    "Defaults to the OS's tempdir.",
)
flags.DEFINE_bool(
    "summarize",
    True,
    "Summarize the inputs and outputs of each module trace logged to disk.",
)
flags.DEFINE_bool(
    "log_all_traces",
    False,
    "Log all traces to logging.info, even if comparison passes.",
)

FLAGS = flags.FLAGS
DEFAULT_INPUT_GENERATOR = tf_utils.uniform


def _setup_artifacts_dir(relative_artifacts_dir: str) -> str:
    parent_dirs = [
        FLAGS.artifacts_dir,
        os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"),
        os.environ.get("TEST_TMPDIR"),
        os.path.join(tempfile.gettempdir(), "iree", "modules"),
    ]
    # Use the most preferred path in parent_dirs that isn't None.
    parent_dir = next(parent for parent in parent_dirs if parent is not None)

    artifacts_dir = os.path.join(parent_dir, relative_artifacts_dir)
    logging.info("Saving compilation artifacts and traces to '%s'", artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


def _parse_target_backends() -> Tuple[Sequence[str], Sequence[str]]:
    """Decodes --target_backends and creates unique ids for them."""
    backend_names = FLAGS.target_backends
    backend_to_index = {k: 0 for k in backend_names if backend_names.count(k) > 1}
    backend_ids = []

    # If there are multiple copies of the same backend_name, index them. e.g.
    # backend_names = ["tf", "iree_vmvx", "tf"]
    # --> backend_ids = ["tf_0", "iree_vmvx", "tf_1"]
    for backend_name in backend_names:
        if backend_name in backend_to_index:
            backend_ids.append(f"{backend_name}_{backend_to_index[backend_name]}")
            backend_to_index[backend_name] += 1
        else:
            backend_ids.append(backend_name)

    return backend_names, backend_ids


def get_target_backends() -> Sequence[module_utils.BackendInfo]:
    """Gets the BackendInfo instances to compare with the reference backend.

    By default all backends in BackendInfo will be used. Specific backends to
    run on can be specified using the `--target_backends` flag.

    Returns:
      Sequence of BackendInfo that should be used.
    """
    if FLAGS.target_backends is not None:
        logging.info("Using backends from command line: %s", FLAGS.target_backends)
        backend_names, backend_ids = _parse_target_backends()
        backends = [
            module_utils.BackendInfo(backend_name, backend_id)
            for backend_name, backend_id in zip(backend_names, backend_ids)
        ]
    else:
        # If no backends are specified, use them all.
        backends = module_utils.BackendInfo.get_all_backends()
    return backends


@dataclass(frozen=True)
class Modules:
    """Compiled modules.

    Args:
      ref_module: Module compiled with the reference backend.
      tar_modules: Sequence of modules compiled with the different target
        backends.
      artifacts_dir: String pointing to where compilation artifacts were saved.
    """

    ref_module: module_utils.CompiledModule
    tar_modules: Sequence[module_utils.CompiledModule]
    artifacts_dir: str


# We have to use a global variable to store the compiled modules so that we can
# avoid recompilation. This is because the TestCase class resets it's entire
# state and calls __init__ before each unit_test. It also calls __init__ one
# additional time before that for good measure, which means without storing the
# modules somewhere else we would have to compile each of them at least twice.
# We can't store the modules on the class itself via setUpClass because of #2900
global _global_modules
_global_modules = None


def compile_tf_module(
    module_class: Type[tf.Module],
    exported_names: Sequence[str] = (),
    relative_artifacts_dir: str = None,
) -> Modules:
    """Compiles module_class to each backend that we test.

    Args:
      module_class: the tf.Module subclass to compile.
      exported_names: optional iterable of strings representing which of
        module_class's functions to compile. If exported_names is empty all
        functions will be compiled.
      relative_artifacts_dir: optional string specifying where to save compilation
        artifacts within the artifacts_dir. If it is not specified then
        module_class.__name__ will be used.

    Returns:
      A 'Modules' dataclass containing the reference module, target modules and
      artifacts directory.
    """
    global _global_modules
    if _global_modules is not None:
        return _global_modules

    # Setup the directory for saving compilation artifacts and traces.
    if relative_artifacts_dir is None:
        relative_artifacts_dir = module_class.__name__
    artifacts_dir = _setup_artifacts_dir(relative_artifacts_dir)

    # Get the backend information for this test.
    ref_backend_info = module_utils.BackendInfo(
        FLAGS.reference_backend, f"{FLAGS.reference_backend}_ref"
    )
    tar_backend_infos = get_target_backends()

    compile_backend = lambda backend_info: backend_info.compile_from_class(
        module_class, exported_names, artifacts_dir
    )

    ref_module = compile_backend(ref_backend_info)
    tar_modules = [compile_backend(backend_info) for backend_info in tar_backend_infos]
    _global_modules = Modules(ref_module, tar_modules, artifacts_dir)
    return _global_modules


def compile_tf_signature_def_saved_model(
    saved_model_dir: str,
    saved_model_tags: Set[str],
    module_name: str,
    exported_name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> Modules:
    """Compiles a SignatureDef SavedModel to each backend that we test.

    Args:
      saved_model_dir: Directory of the saved model.
      saved_model_tags: Optional set of tags to use when loading the model.
      module_name: A name for this compiled module.
      backend_info: BackendInfo with the details for compiling the saved model.
      exported_name: A str representing the signature on the saved model to
        compile.
      input_names: A sequence of kwargs to feed to the saved model.
      output_names: A sequence of named outputs to extract from the saved model.

    Returns:
      A 'Modules' dataclass containing the reference module, target modules and
      artifacts directory.
    """
    global _global_modules
    if _global_modules is not None:
        return _global_modules

    # Setup the directory for saving compilation artifacts and traces.
    artifacts_dir = _setup_artifacts_dir(module_name)

    # Get the backend information for this test.
    ref_backend_info = module_utils.BackendInfo(
        FLAGS.reference_backend, f"{FLAGS.reference_backend}_ref"
    )
    tar_backend_infos = get_target_backends()

    compile_backend = (
        lambda backend_info: backend_info.compile_signature_def_saved_model(
            saved_model_dir,
            saved_model_tags,
            module_name,
            exported_name,
            input_names,
            output_names,
            artifacts_dir,
        )
    )

    ref_module = compile_backend(ref_backend_info)
    tar_modules = [compile_backend(backend_info) for backend_info in tar_backend_infos]
    _global_modules = Modules(ref_module, tar_modules, artifacts_dir)
    return _global_modules


# We use global variables to store the configuration information for
# tf_function_unit_tests because tensorflow.python.eager.def_function.Function
# is not an API that we can subclass, and storing the information directly
# that class results in it being deleted at tf.Module initialization.
# _global_unit_test_configs is a dict mapping exported_names to dicts containing
# a get-function for input args and the tolerance kwargs for the trace.
global _global_unit_test_configs
_global_unit_test_configs = dict()


class UnitTestSpec:
    def __init__(
        self,
        unit_test_name: str,
        input_signature: Sequence[tf.TensorSpec],
        input_generator: tf_utils.InputGeneratorType = None,
        input_args: Union[Sequence[Any], None] = None,
        kwargs: Dict[str, Any] = None,
    ):
        self.unit_test_name = tf_utils.remove_special_characters(unit_test_name)
        self.input_signature = input_signature
        self.input_args = input_args
        self.kwargs = dict() if kwargs is None else kwargs
        self.input_generator = input_generator

    def with_name(self, new_name: str) -> UnitTestSpec:
        return UnitTestSpec(
            new_name,
            self.input_signature,
            self.input_generator,
            self.input_args,
            self.kwargs,
        )

    def __str__(self):
        return self.unit_test_name


def _dictionary_product(dictionary: Dict[Any, Any]) -> List[Dict[Any, Any]]:
    """Returns a named cartesian product of dictionary's values.

    Converts {'a': [1, 2], 'b': [3, 4]} into
    [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    product = [[]]
    for values in dictionary.values():
        # Iteratively grow the elements of the product.
        product = [element + [value] for element in product for value in values]
    dicts = [{k: v for k, v in zip(dictionary, element)} for element in product]
    return dicts


def _named_kwargs_product(
    kwargs_to_values: Dict[str, Sequence[Any]]
) -> Dict[str, Dict[str, Any]]:
    """Splits kwargs_to_values into a Cartesian product of its elements."""
    # Validate 'kwargs_to_values'
    if kwargs_to_values is None:
        kwargs_to_values = dict()  # Use only default kwargs.
    for kwarg_key, kwarg_values in kwargs_to_values.items():
        if not isinstance(kwarg_values, Sequence):
            raise TypeError(
                f"Expected kwargs_to_values[{repr(kwarg_key)}] to be a "
                f"sequence, but got '{type(kwarg_values)}'"
            )

    # Expand across a Cartesian product.
    kwargs_product = _dictionary_product(kwargs_to_values)
    # {'a': 1, 'b': 3} -> "a_1__b_3"
    dict_to_str = lambda d: "__".join([f"{k}_{v}" for k, v in d.items()])
    return {dict_to_str(kwargs): kwargs for kwargs in kwargs_product}


def unit_test_specs_from_signatures(
    signature_shapes: Sequence[Sequence[Sequence[int]]],
    signature_dtypes: Sequence[tf.DType] = [tf.float32],
    input_generators: Union[
        Sequence[tf_utils.InputGeneratorType], Dict[str, tf_utils.InputGeneratorType]
    ] = [DEFAULT_INPUT_GENERATOR],
    kwargs_to_values: Dict[str, Sequence[Any]] = None,
) -> List[UnitTestSpec]:
    """Generates a Cartesian product of UnitTestSpecs from the given arguments.

    Args:
      signature_shapes:
        A sequence (representing multiple signatures to test) of sequences
        (representing the shapes of the args in those signatures) of ints
        (representing the individual sizes of those shapes).
      signature_dtypes:
        A sequence of dtypes to test each signature with.
      input_generators:
        Either:
          1. a sequence of input generators to test each of the signature-dtype
             pairs with
          2. a dictionary mapping input generator names to input generators to
             test each of the signature-dtype pairs with. This format must be used
             if any of the generators are lambda functions.
      kwargs_to_values:
        A dict mapping kwarg names to sequences of values that they can take.

    Returns:
      A list of 'UnitTestSpec's generated from the provided arguments.
    """
    # Validate 'signature_shapes'
    for i, shapes in enumerate(signature_shapes):
        if not isinstance(shapes, Sequence):
            raise TypeError(
                f"Expected signature_shapes[{i}] to be a sequence, but "
                f"got '{type(shapes)}'"
            )
        for j, shape in enumerate(shapes):
            if not isinstance(shape, Sequence):
                raise TypeError(
                    f"Expected signature_shapes[{i}][{j}] to be a "
                    f"sequence, but got '{type(shape)}'"
                )
            for k, size in enumerate(shape):
                if not isinstance(size, int):
                    raise TypeError(
                        f"Expected signature_shapes[{i}][{j}][{k}] to be an "
                        f"int but got '{type(size)}"
                    )

    # Parse 'signature_shapes'
    names_to_shapes = dict()
    for signature in signature_shapes:
        # Converts [[1, 2, 3], [4, 5]] into 1x2x3_4x5.
        signature_key = "_".join(
            ["x".join(str(size) for size in shape) for shape in signature]
        )
        names_to_shapes[signature_key] = signature

    # Validate 'signature_dtypes'
    for i, dtype in enumerate(signature_dtypes):
        if not isinstance(dtype, tf.DType):
            raise TypeError(
                f"Expected dtypes[{i}] to be a tf.DType, but got '{type(dtype)}'"
            )

    # Parse 'signature_dtypes'
    # 'complex64' -> 'c64'
    abbreviate = lambda dtype: re.sub(r"([a-z])[a-z]*([0-9]+)", r"\1\2", dtype)
    names_to_dtypes = {abbreviate(dtype.name): dtype for dtype in signature_dtypes}

    # Validate 'input_generators'
    if not isinstance(input_generators, (Sequence, Dict)):
        raise TypeError(
            "Expected 'input_generators' to be a sequence or "
            f"dictionary, but got '{type(input_generators)}'"
        )
    if isinstance(input_generators, Sequence):
        for i, generator in enumerate(input_generators):
            if generator.__name__ == "<lambda>":
                raise TypeError(
                    f"'input_generators' was a sequence but input_generators[{i}] was "
                    "lambda function. 'input_generators' must be a dictionary if "
                    "lambda functions are used."
                )

    # Parse 'input_generators'
    if isinstance(input_generators, Sequence):
        names_to_generators = {gen.__name__: gen for gen in input_generators}
    else:
        names_to_generators = input_generators

    # Validate and parse 'kwargs_to_values'
    names_to_kwargs = _named_kwargs_product(kwargs_to_values)

    # Create a Cartesian product through all specifications and their names.
    specs = [names_to_shapes, names_to_dtypes, names_to_generators, names_to_kwargs]
    # pytype: disable=attribute-error
    key_product = itertools.product(*[list(spec.keys()) for spec in specs])
    value_product = itertools.product(*[list(spec.values()) for spec in specs])
    # pytype: enable=attribute-error

    # Generate a UnitTestSpec for each element in the above product.
    unit_tests = []
    for keys, (shapes, dtype, generator, kwargs) in zip(key_product, value_product):
        unit_test_name = "__".join(key for key in keys if key)
        input_signature = [tf.TensorSpec(shape, dtype) for shape in shapes]
        unit_tests.append(
            UnitTestSpec(
                unit_test_name=unit_test_name,
                input_signature=input_signature,
                input_generator=generator,
                input_args=None,
                kwargs=kwargs,
            )
        )
    return unit_tests


def unit_test_specs_from_args(
    names_to_input_args: Dict[str, Sequence[Any]],
    kwargs_to_values: Dict[str, Sequence[Any]] = None,
) -> List[UnitTestSpec]:
    """Generates a Cartesian product of UnitTestSpecs from the given arguments.

    Args:
      signature_shapes:
        A dict mapping names for input arguments to the arguments themselves.
      kwargs_to_values:
        A dict mapping kwarg names to sequences of values that they can take.

    Returns:
      A list of 'UnitTestSpec's generated from the provided arguments.
    """
    # Validate and parse 'kwargs_to_values'
    names_to_kwargs = _named_kwargs_product(kwargs_to_values)

    # Create a Cartesian product through all specifications and their names.
    specs = [names_to_input_args, names_to_kwargs]
    key_product = itertools.product(*[list(spec.keys()) for spec in specs])
    value_product = itertools.product(*[list(spec.values()) for spec in specs])

    # Generate a UnitTestSpec for each element in the above product.
    unit_tests = []
    for keys, (input_args, kwargs) in zip(key_product, value_product):
        unit_test_name = "__".join(key for key in keys if key)
        input_signature = tf_utils.apply_function(
            input_args, lambda x: tf.TensorSpec.from_tensor(tf.convert_to_tensor(x))
        )
        unit_tests.append(
            UnitTestSpec(
                unit_test_name=unit_test_name,
                input_signature=input_signature,
                input_generator=None,
                input_args=input_args,
                kwargs=kwargs,
            )
        )
    return unit_tests


def tf_function_unit_test(
    input_generator: tf_utils.InputGeneratorType = None,
    input_args: Sequence[Any] = None,
    atol: float = None,
    rtol: float = None,
    name: str = None,
    static_signature: Sequence[tf.TensorSpec] = None,
    **tf_function_kwargs,
):
    """Creates a tf.function that can be used to generate unit_tests.

    If 'input_generator' and 'input_args' are unspecified then the function will
    be tested using random uniform data.

    Args:
      input_generator:
        an optional callable taking a shape and dtype that returns input data for
        the unit_test.
      input_args:
        an optional sequence of values to pass as positional args to the function.
      atol:
        optional, the absolute tolerance to use when comparing the decorated
        function's output.
      rtol:
        optional, the relative tolerance to use when comparing the decorated
        function's output.
      name:
        optional, the name to reference this function with. Must be used if
        decorating a lambda.
      static_signature:
        optional, a signature with the same structure as 'input_signature'. Used
        to specify the correct shape for data generation when dynamic dims are
        provided.

    Raises:
      ValueError: if 'input_generator' and 'input_args' are both specified.

    Returns:
      A tf.function with the additional attributes 'input_generator' (from above)
      'trace_kwargs' (from 'atol' and 'rtol' above), and with an updated
      __name__ attribute if 'name' was specified.
    """

    def _store_unit_test_info(function):
        # Validate arguments.
        if input_generator is not None and input_args is not None:
            raise ValueError(
                "'input_generator' and 'input_args' cannot both be specified."
            )

        function = tf.function(**tf_function_kwargs)(function)

        # Set function.__name__
        if name is not None:
            function.__name__ = name
        elif function.__name__ == "<lambda>":
            raise ValueError(
                "The 'name' kwarg must be provided when decorating a "
                "lambda function."
            )

        global _global_unit_test_configs
        if function.__name__ not in _global_unit_test_configs:
            if static_signature is not None:
                signature = static_signature
            else:
                signature = function.input_signature

            if input_generator is not None:
                # Use the user-specificed input_generator.
                get_trace_args = lambda: tf_utils.generate_inputs(
                    signature, input_generator
                )
            elif input_args is not None:
                # Use the user-specified input_args.
                get_trace_args = lambda: copy.deepcopy(input_args)
            else:
                # No user data specification – default to using random uniform data.
                get_trace_args = lambda: tf_utils.generate_inputs(
                    signature, DEFAULT_INPUT_GENERATOR
                )

            _global_unit_test_configs[function.__name__] = dict(
                get_trace_args=get_trace_args, trace_kwargs=dict(atol=atol, rtol=rtol)
            )

        return function

    return _store_unit_test_info


class TestModule(tf.Module):
    """Thin tf.Module wrapper with helper methods for tf_function_unit_tests."""

    @classmethod
    def get_tf_function_unit_tests(cls):
        """Get all tf_function_unit_test-created tf.functions on the class."""
        # Initialize the module to ensure that _global_unit_test_configs has the
        # info for all of the unit_tests. (Only doing this if
        # _global_unit_test_configs is empty wouldn't address the case where some
        # unit_tests are defined on the class and some are generated by __init__).
        cls()

        tf_function_unit_tests = list(_global_unit_test_configs.keys())
        if not len(tf_function_unit_tests):
            raise ValueError(
                "'get_tf_function_unit_tests' was called but no tests were found."
            )
        return tf_function_unit_tests


class TracedModuleTestCase(tf.test.TestCase):
    """Compiles a tf.Module to multiple backends to test their correctness."""

    def setUp(self) -> None:
        # Runs before each unit test.
        super().setUp()
        self._modules.ref_module.reinitialize()
        for module in self._modules.tar_modules:
            module.reinitialize()

    @classmethod
    def generate_unit_tests(cls, module_class: Type[TestModule]):
        """Generates tests for each 'tf_function_unit_test' on 'module_class'."""
        for function_name in module_class.get_tf_function_unit_tests():
            # We have to pass the closure arguments 'function_name', 'get_args' and
            # 'kwargs' to 'trace' via a kwarg instead of using it directly in the body
            # because 'function_name' and 'unit_test_config' are overwritten in each
            # iteration of this loop, and python will only use the most recent version
            # of each. If we didn't do this, then we would only test the last function
            # in this loop. The same is true for passing 'trace' to 'unit_test'.
            unit_test_config = _global_unit_test_configs[function_name]

            # Runs the inputs through a (traced) module.
            def trace(
                module,
                function_name=function_name,
                get_args=unit_test_config["get_trace_args"],
                kwargs=unit_test_config["trace_kwargs"],
            ):
                getattr(module, function_name)(*get_args(), **kwargs)

            # Give the trace the name of the tf.function that it is testing.
            trace.__name__ = function_name

            # Runs 'trace' on modules compiled to each backend and compares them.
            def unit_test(self, trace=trace):
                self.compare_backends(trace, self._modules)

            # Make 'unit_test' a function on the TracedModuleTestCase, which tells
            # the test runner to run it.
            unit_test.__name__ = f"test_{function_name}"
            if hasattr(cls, unit_test.__name__):
                raise ValueError(
                    "Tried to generate multiple instances of the "
                    f"unit_test '{unit_test.__name__}'."
                )
            setattr(cls, unit_test.__name__, unit_test)

    def compare_backends(
        self,
        trace_function: Callable[[trace_utils.TracedModule], None],
        modules: Modules,
    ) -> None:
        """Run the reference and target backends on trace_function and compare them.

        Random seeds for tensorflow, numpy and python are set before each invocation
        of trace_function.

        Args:
          trace_function: a function accepting a TracedModule as its argument.
        """
        # Create Traces for each backend.
        ref_trace = trace_utils.Trace(modules.ref_module, trace_function)
        tar_traces = [
            trace_utils.Trace(module, trace_function) for module in modules.tar_modules
        ]

        # Run the traces through trace_function with their associated modules.
        tf_utils.set_random_seed()
        trace_function(trace_utils.TracedModule(modules.ref_module, ref_trace))
        if FLAGS.log_all_traces:
            logging.info(ref_trace)
        for module, trace in zip(modules.tar_modules, tar_traces):
            tf_utils.set_random_seed()
            trace_function(trace_utils.TracedModule(module, trace))
            if FLAGS.log_all_traces:
                logging.info(trace)

        # Compare each target trace of trace_function with the reference trace.
        failed_backend_indices = []
        error_messages = []
        for i, tar_trace in enumerate(tar_traces):
            logging.info(
                "Comparing the reference backend '%s' with '%s'",
                ref_trace.backend_id,
                tar_trace.backend_id,
            )
            traces_match, errors = trace_utils.compare_traces(ref_trace, tar_trace)
            if not traces_match:
                failed_backend_indices.append(i)
                error_messages.extend(errors)

        # Save the results to disk before validating.
        ref_trace_dir = trace_utils.get_trace_dir(modules.artifacts_dir, ref_trace)
        ref_trace.save_plaintext(ref_trace_dir, FLAGS.summarize)
        ref_trace.serialize(ref_trace_dir)
        for tar_trace in tar_traces:
            tar_trace_dir = trace_utils.get_trace_dir(modules.artifacts_dir, tar_trace)
            tar_trace.save_plaintext(tar_trace_dir, FLAGS.summarize)
            tar_trace.serialize(tar_trace_dir)

        # Validate results.
        if failed_backend_indices:
            # Extract info for logging.
            failed_backends = [tar_traces[i].backend_id for i in failed_backend_indices]
            error_list = "".join([f"\n  - {message}" for message in error_messages])
            self.fail(
                "Comparison between the reference backend and the following targets "
                f"failed: {failed_backends}. Errors: {error_list}\n"
                "See the logs above for more details about the non-matching calls."
            )

    @classmethod
    def tearDownClass(cls) -> None:
        # Runs after all unit tests are completed.
        super().tearDownClass()
