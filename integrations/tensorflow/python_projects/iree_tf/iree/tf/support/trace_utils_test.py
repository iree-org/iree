# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for iree.tf.support.trace_utils."""

import os
import tempfile

from absl.testing import parameterized
from iree.tf.support import module_utils
from iree.tf.support import trace_utils
import numpy as np
import tensorflow as tf


class StatefulCountingModule(tf.Module):
    def __init__(self):
        self.count = tf.Variable([0.0])

    @tf.function(input_signature=[])
    def increment(self):
        self.count.assign_add(tf.constant([1.0]))

    @tf.function(input_signature=[])
    def get_count(self):
        return self.count

    @tf.function(input_signature=[tf.TensorSpec([1])])
    def increment_by(self, value):
        self.count.assign_add(value)

    @tf.function(input_signature=[tf.TensorSpec([1]), tf.TensorSpec([1])])
    def increment_by_max(self, a, b):
        result = tf.maximum(a, b)
        self.count.assign_add(result)
        return result

    @tf.function(input_signature=[])
    def decrement(self):
        self.count.assign_sub(tf.constant([1.0]))


class TestUtilsTests(tf.test.TestCase, parameterized.TestCase):
    def test_trace_inputs_and_outputs(self):
        def trace_function(module):
            # No inputs or outputs
            module.increment()
            # Only inputs
            module.increment_by(np.array([81.0], dtype=np.float32))
            # Only outputs
            module.get_count()

        module = module_utils.TfCompiledModule.create_from_class(
            StatefulCountingModule, module_utils.BackendInfo("tf")
        )
        trace = trace_utils.Trace(module, trace_function)
        trace_function(trace_utils.TracedModule(module, trace))

        self.assertIsInstance(trace.calls[0].inputs, tuple)
        self.assertEmpty(trace.calls[0].inputs)
        self.assertIsInstance(trace.calls[0].outputs, tuple)
        self.assertEmpty(trace.calls[0].outputs)

        self.assertAllClose(trace.calls[1].inputs[0], [81.0])
        self.assertAllClose(trace.calls[2].outputs[0], [82.0])

    def test_nonmatching_methods(self):
        def tf_function(module):
            module.increment()
            module.increment()

        def vmvx_function(module):
            module.increment()
            module.decrement()

        tf_module = module_utils.TfCompiledModule.create_from_class(
            StatefulCountingModule, module_utils.BackendInfo("tf")
        )
        tf_trace = trace_utils.Trace(tf_module, tf_function)
        tf_function(trace_utils.TracedModule(tf_module, tf_trace))

        vmvx_module = module_utils.IreeCompiledModule.create_from_class(
            StatefulCountingModule, module_utils.BackendInfo("iree_vmvx")
        )
        vmvx_trace = trace_utils.Trace(vmvx_module, vmvx_function)
        vmvx_function(trace_utils.TracedModule(vmvx_module, vmvx_trace))

        with self.assertRaises(ValueError):
            trace_utils.compare_traces(tf_trace, vmvx_trace)

    def test_nonmatching_inputs(self):
        def tf_function(module):
            module.increment_by(np.array([42.0], dtype=np.float32))

        def vmvx_function(module):
            module.increment_by(np.array([22.0], dtype=np.float32))

        tf_module = module_utils.TfCompiledModule.create_from_class(
            StatefulCountingModule, module_utils.BackendInfo("tf")
        )
        tf_trace = trace_utils.Trace(tf_module, tf_function)
        tf_function(trace_utils.TracedModule(tf_module, tf_trace))

        vmvx_module = module_utils.IreeCompiledModule.create_from_class(
            StatefulCountingModule, module_utils.BackendInfo("iree_vmvx")
        )
        vmvx_trace = trace_utils.Trace(vmvx_module, vmvx_function)
        vmvx_function(trace_utils.TracedModule(vmvx_module, vmvx_trace))

        same, error_messages = trace_utils.compare_traces(tf_trace, vmvx_trace)
        self.assertFalse(same)

    def test_trace_serialize_and_load(self):
        def trace_function(module):
            module.increment()
            module.increment_by(np.array([81.0], dtype=np.float32))
            module.increment_by_max(
                np.array([81], dtype=np.float32), np.array([92], dtype=np.float32)
            )
            module.get_count()

        module = module_utils.IreeCompiledModule.create_from_class(
            StatefulCountingModule, module_utils.BackendInfo("iree_vmvx")
        )
        trace = trace_utils.Trace(module, trace_function)
        trace_function(trace_utils.TracedModule(module, trace))

        with tempfile.TemporaryDirectory() as artifacts_dir:
            trace_function_dir = trace_utils.get_trace_dir(artifacts_dir, trace)
            trace.serialize(trace_function_dir)
            self.assertTrue(
                os.path.exists(os.path.join(trace_function_dir, "metadata.pkl"))
            )
            loaded_trace = trace_utils.Trace.load(trace_function_dir)

            # Check all calls match.
            self.assertTrue(trace_utils.compare_traces(trace, loaded_trace))

            # Check all other metadata match.
            self.assertAllEqual(trace.__dict__.keys(), loaded_trace.__dict__.keys())
            for key in trace.__dict__.keys():
                if key != "calls":
                    self.assertEqual(trace.__dict__[key], loaded_trace.__dict__[key])


if __name__ == "__main__":
    tf.test.main()
