# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for tracing tf.function inputs and outputs."""

# This file uses the following abbreviations:
#   ref: reference – for the reference CompiledModule
#   tar: target - for one of the target CompiledModules

from __future__ import annotations
import copy
import glob
import inspect
import os
import pickle
import sys
import textwrap
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Optional

from absl import logging
from iree.tf.support import module_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf

NUMPY_LINEWIDTH = 120
INDENT = " " * 2


def _zfill_width(length: int) -> Union[int, None]:
    return int(np.ceil(np.log10(length))) if length else None


def get_trace_dir(artifacts_dir: str, trace: Trace) -> str:
    trace_dir = os.path.join(
        artifacts_dir, trace.backend_id, "traces", trace.function_name
    )
    os.makedirs(trace_dir, exist_ok=True)
    return trace_dir


class ModuleCall:
    def __init__(
        self,
        method: str,
        inputs: Tuple[Any],
        outputs: Tuple[Any],
        serialized_inputs: Tuple[str],
        serialized_outputs: Tuple[str],
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        """Records the details of a call to a CompiledModule."""
        self.method = method

        # Deepcopy to safegard against mutation.
        self.inputs = copy.deepcopy(inputs)
        if outputs is not None:
            outputs = copy.deepcopy(outputs)
        else:
            outputs = tuple()
        self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)

        self.serialized_inputs = serialized_inputs
        self.serialized_outputs = serialized_outputs

        self.rtol = rtol
        self.atol = atol

    def get_tolerances(self) -> Tuple[float, float]:
        """Gets the floating point tolerances associated with this call."""
        return self.rtol, self.atol

    def _get_shape_and_dtype(self, value: Any) -> str:
        if isinstance(value, np.ndarray):
            return tf_utils.get_shape_and_dtype(value, allow_non_mlir_dtype=True)
        else:
            return str(type(value))

    def __str__(self):
        prior_printoptions = np.get_printoptions()
        np.set_printoptions(linewidth=NUMPY_LINEWIDTH)

        header = f"Method: {self.method}"
        inputs = "\n".join(
            [textwrap.indent(str(value), INDENT) for value in self.inputs]
        )
        input_shapes = ", ".join(
            self._get_shape_and_dtype(value) for value in self.inputs
        )

        outputs = "\n".join(
            [textwrap.indent(str(value), INDENT) for value in self.outputs]
        )
        output_shapes = ", ".join(
            self._get_shape_and_dtype(value) for value in self.outputs
        )

        tolerances = textwrap.indent(f"rtol={self.rtol}, atol={self.atol}", INDENT)
        body = (
            f"Inputs: {input_shapes}\n{inputs}\n"
            f"Outputs: {output_shapes}\n{outputs}"
            f"\nTolerances:\n{tolerances}"
        )
        result = f"{header}\n{textwrap.indent(body, INDENT)}"

        np.set_printoptions(**prior_printoptions)
        return result

    def serialize(self, call_dir: str) -> None:
        """Stores a serialized copy of this call.

        Can be loaded via ModuleCall.load(call_dir)

        Args:
          call_dir: str, the path to the directory to serialize this call to.
        """
        os.makedirs(call_dir, exist_ok=True)

        metadata = {
            "method": self.method,
            "serialized_inputs": self.serialized_inputs,
            "serialized_outputs": self.serialized_outputs,
            "rtol": self.rtol,
            "atol": self.atol,
        }
        with open(os.path.join(call_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        width = _zfill_width(len(self.inputs))
        for i, value in enumerate(self.inputs):
            path = os.path.join(call_dir, f"input_{str(i).zfill(width)}.pkl")
            with open(path, "wb") as f:
                pickle.dump(value, f)

        width = _zfill_width(len(self.outputs))
        for i, value in enumerate(self.outputs):
            path = os.path.join(call_dir, f"output_{str(i).zfill(width)}.pkl")
            with open(path, "wb") as f:
                pickle.dump(value, f)

    @staticmethod
    def load(call_dir: str) -> ModuleCall:
        """Loads and returns a trace serialized with ModuleCall.serialize."""
        with open(os.path.join(call_dir, "metadata.pkl"), "rb") as f:
            kwargs = pickle.load(f)

        for result_type in ["input", "output"]:
            key = f"{result_type}s"  # inputs or outputs
            kwargs[key] = []

            files = glob.glob(os.path.join(call_dir, f"{result_type}_*.pkl"))
            for filename in sorted(files):
                with open(filename, "rb") as f:
                    kwargs[key].append(pickle.load(f))

            # Convert to tuple to match python's return type for multiple results.
            kwargs[key] = tuple(kwargs[key])

        return ModuleCall(**kwargs)


class Trace:
    """Stores the inputs and outputs of a series of calls to a module."""

    def __init__(
        self,
        module: Union[module_utils.CompiledModule, None],
        function: Union[Callable[[TracedModule], None], None],
        _load_dict: Optional[Dict[str, Any]] = None,
    ):
        """Extracts metadata from module and function and initializes.

        Example usage:
          def forward_pass(...):
            ...
          module = IreeCompiledModule(...)
          trace = Trace(module, forward_pass)
          forward_pass(TracedModule(module, trace))

        Args:
          module: the module who's outputs this trace will record.
          function: the function that module will be traced on.
          _load_dict: used internally
        """
        if _load_dict is None:
            # Extract metadata from module and function.
            self.module_name = module.module_name
            self.compiled_paths = module.compiled_paths
            self.backend_name = module.backend_info.backend_name
            self.backend_id = module.backend_info.backend_id
            self.backend_driver = module.backend_info.driver
            self.iree_serializable = module.iree_serializable()
            self.tflite_serializable = module.tflite_serializable()
            self.function_name = function.__name__
            self.function_sourcefile = inspect.getsourcefile(function)
            source, start_line = inspect.getsourcelines(function)
            self.function_line_numbers = (start_line, start_line + len(source))
            self.function_source = "".join(source)

            self.calls = []
        else:
            self.module_name = _load_dict["module_name"]
            self.compiled_paths = _load_dict["compiled_paths"]
            self.backend_name = _load_dict["backend_name"]
            self.backend_id = _load_dict["backend_id"]
            self.backend_driver = _load_dict["backend_driver"]
            self.iree_serializable = _load_dict["iree_serializable"]
            self.tflite_serializable = _load_dict["tflite_serializable"]
            self.function_name = _load_dict["function_name"]
            self.function_sourcefile = _load_dict["function_sourcefile"]
            self.function_line_numbers = _load_dict["function_line_numbers"]
            self.function_source = _load_dict["function_source"]
            self.calls = _load_dict["calls"]

    def __str__(self):
        header = (
            f"Trace of {self.module_name} compiled to '{self.backend_id}' "
            f"on function '{self.function_name}':"
        )
        # Give each call a number so it's easier to compare between multiple traces.
        calls = [f"{i + 1}. {str(call)}" for i, call in enumerate(self.calls)]
        calls = textwrap.indent("\n".join(calls), prefix=INDENT)
        return f"{header}\n{calls}"

    def __iter__(self):
        for call in self.calls:
            yield call

    def save_plaintext(self, trace_dir: str, summarize: bool = True) -> None:
        """Saves a human-readable string representation of this trace to disk.

        Args:
          trace_dir: str, path to the directory to save the trace in.
          summarize: a bool controlling whether numpy should summarize the inputs
            and outputs if they're large. Setting this to False is very slow for
            large outputs.
        """
        prior_printoptions = np.get_printoptions()
        np.set_printoptions(
            linewidth=NUMPY_LINEWIDTH,
            threshold=None if summarize else sys.maxsize,
            edgeitems=10,
        )  # Can show more items since they won't clutter the logs.

        path = os.path.join(trace_dir, "log.txt")
        with open(path, "w") as f:
            f.write(str(self))
            f.write("\n")

        np.set_printoptions(**prior_printoptions)

    def serialize(self, trace_dir: str) -> None:
        """Stores a serialized copy of this trace in trace_dir.

        It can be loaded via `Trace.load(trace_dir)`.

        Args:
          trace_dir: str, path to the directory to serialize the trace to.
        """

        compiled_paths = None
        if self.compiled_paths is not None:
            # Convert to a dict to avoid the issues with serializing defaultdicts.
            compiled_paths = dict(self.compiled_paths)

        # Python serialization.
        metadata = {
            "module_name": self.module_name,
            "compiled_paths": compiled_paths,
            "backend_name": self.backend_name,
            "backend_id": self.backend_id,
            "backend_driver": self.backend_driver,
            "iree_serializable": self.iree_serializable,
            "tflite_serializable": self.tflite_serializable,
            "function_name": self.function_name,
            "function_sourcefile": self.function_sourcefile,
            "function_line_numbers": self.function_line_numbers,
            "function_source": self.function_source,
        }
        with open(os.path.join(trace_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        width = _zfill_width(len(self.calls))
        for i, call in enumerate(self.calls):
            call_dir = os.path.join(trace_dir, f"call_{str(i).zfill(width)}")
            call.serialize(call_dir)

        # C++ benchmark serialization.
        if self.iree_serializable or self.tflite_serializable:
            entry_function = self.calls[0].method
            compiled_path = self.compiled_paths[entry_function]

            if self.iree_serializable:
                serialized_inputs = self.calls[0].serialized_inputs
                flagfile = [
                    f"--module={compiled_path}",
                    f"--device={self.backend_driver}",
                    f"--function={entry_function}",
                ] + [f'--input="{input}"' for input in serialized_inputs]
                with open(os.path.join(trace_dir, "flagfile"), "w") as f:
                    f.writelines(line + "\n" for line in flagfile)
            else:
                with open(os.path.join(trace_dir, "graph_path"), "w") as f:
                    f.writelines(compiled_path + "\n")

    @staticmethod
    def load(trace_dir: str) -> Trace:
        """Loads and returns a trace serialized with Trace.serialize.

        Args:
          trace_dir: str, path to the directory of the serialized trace.

        Returns:
          A Trace deserialized from trace_dir.
        """
        with open(os.path.join(trace_dir, "metadata.pkl"), "rb") as f:
            load_dict = pickle.load(f)
        call_dirs = sorted(glob.glob(os.path.join(trace_dir, "call_*")))
        calls = [ModuleCall.load(call_dir) for call_dir in call_dirs]
        load_dict["calls"] = calls
        return Trace(module=None, function=None, _load_dict=load_dict)


class TracedModule:
    def __init__(self, module: module_utils.CompiledModule, trace: Trace):
        """Wraps a CompiledModule so that all inputs and outputs are traced.

        The TracedModule returned will have an API almost identical to that of the
        passed CompiledModule. The only changes is that if the keywords `rtol` or
        `atol` are passed to one of the CompiledModule's methods, then they will be
        used to set the tolerance for comparing that call to the same call in
        another trace. So for example, calling `traced_module.add(a, b rtol=1e-8)`
        would be the same as calling `module.add(a, b)`.

        Args:
          module: the CompiledModule to trace.
          trace: the Trace to record calls to this module with.
        """
        self._module = module
        self._trace = trace

    def _trace_call(self, method: module_utils._FunctionWrapper, method_name: str):
        """Decorates a CompiledModule method to capture its inputs and outputs."""

        def call(*args, **kwargs):
            # Pop manually specified tolerances from the kwargs (if any).
            tolerances = {}
            tolerances["rtol"] = kwargs.pop("rtol", None)
            tolerances["atol"] = kwargs.pop("atol", None)
            # Only pass these to ModuleCall if they were specified by the user.
            tolerances = {k: v for k, v in tolerances.items() if v is not None}

            # Ensure the inputs are numpy inputs.
            args = tf_utils.convert_to_numpy(args)
            kwargs = tf_utils.convert_to_numpy(kwargs)

            # Run the method and record the details of the call.
            outputs = method(*args, **kwargs)
            serialized_inputs, serialized_outputs = method.get_serialized_values()
            self._trace.calls.append(
                ModuleCall(
                    method_name,
                    args,
                    outputs,
                    serialized_inputs,
                    serialized_outputs,
                    **tolerances,
                )
            )
            return outputs

        return call

    def __getattr__(self, attr):
        # Try to resolve it as an attr on self._module.
        if not hasattr(self._module, attr):
            raise AttributeError(f"The compiled module does not have attr '{attr}'")
        module_attr = getattr(self._module, attr)
        if not hasattr(module_attr, "__call__"):
            # e.g. traced_module.backend
            return module_attr
        else:
            # e.g. traced_module.simple_mul(a, b)
            return self._trace_call(module_attr, method_name=attr)


def compare_traces(ref_trace: Trace, tar_trace: Trace) -> Tuple[bool, Sequence[str]]:
    traces_match = True
    error_messages = []

    # Check that all method invocations match.
    ref_methods = [(call.method, call.rtol, call.atol) for call in ref_trace]
    tar_methods = [(call.method, call.rtol, call.atol) for call in tar_trace]
    if ref_methods != tar_methods:
        # Raise a ValueError instead of returning False since this is an
        # unexpected error.
        raise ValueError(
            "The reference and target traces have different call structures:\n"
            f"Reference: {ref_methods}\nTarget:    {tar_methods}"
        )

    for ref_call, tar_call in zip(ref_trace, tar_trace):
        logging.info("Comparing calls to '%s'", ref_call.method)
        rtol, atol = ref_call.get_tolerances()

        inputs_match, error_message = tf_utils.check_same(
            ref_call.inputs, tar_call.inputs, rtol, atol
        )
        if not inputs_match:
            error_messages.append(error_message)
            logging.error("Inputs did not match.")
        outputs_match, error_message = tf_utils.check_same(
            ref_call.outputs, tar_call.outputs, rtol, atol
        )
        if not outputs_match:
            error_messages.append(error_message)
            logging.error("Outputs did not match.")
        calls_match = inputs_match and outputs_match

        if not calls_match:
            logging.error(
                "Comparision between '%s' and '%s' failed on method '%s'",
                ref_trace.backend_id,
                tar_trace.backend_id,
                ref_call.method,
            )
            logging.error("Reference call '%s':\n%s", ref_trace.backend_id, ref_call)
            logging.error("Target call '%s':\n%s", tar_trace.backend_id, tar_call)

        traces_match = traces_match and calls_match
    return traces_match, error_messages
