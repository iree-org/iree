# Lint as: python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test architecture for a set of tflite tests."""

import absl
from absl.flags import FLAGS
import absl.testing as testing
import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy as np
import os
import sys
import tempfile
import tensorflow.compat.v2 as tf
import time
import urllib.request

targets = {
    'dylib': 'dylib-llvm-aot',
    'vulkan': 'vulkan-spirv',
}

configs = {
    'dylib': 'dylib',
    'vulkan': 'vulkan',
}

absl.flags.DEFINE_string('config', 'dylib', 'model path to execute')


class TFLiteModelTest(testing.absltest.TestCase):

  def __init__(self, model_path, *args, **kwargs):
    super(TFLiteModelTest, self).__init__(*args, **kwargs)
    self.model_path = model_path

  def setUp(self):
    if self.model_path is None:
      return
    exe_basename = os.path.basename(sys.argv[0])
    self.workdir = tempfile.mkdtemp(dir=testing.absltest.TEST_TMPDIR.value)
    print(f"TMP_DIR = {self.workdir}")
    self.tflite_file = '/'.join([self.workdir, 'model.tflite'])
    self.tflite_ir = '/'.join([self.workdir, 'tflite.mlir'])
    self.iree_ir = '/'.join([self.workdir, 'tosa.mlir'])
    if os.path.exists(self.model_path):
      self.tflite_file = self.model_path
    else:
      urllib.request.urlretrieve(self.model_path, self.tflite_file)
    self.binary = '/'.join([self.workdir, 'module.bytecode'])

  def generate_inputs(self, input_details):
    args = []
    for input in input_details:
      absl.logging.info("\t%s, %s", str(input["shape"]),
                        input["dtype"].__name__)
      args.append(np.zeros(shape=input["shape"], dtype=input["dtype"]))
    return args

  def compare_results(self, iree_results, tflite_results, details):
    self.assertEqual(len(iree_results), len(tflite_results),
                     "Number of results do not match")

    for i in range(len(details)):
      iree_result = iree_results[i]
      tflite_result = tflite_results[i]
      iree_result = iree_result.astype(np.single)
      tflite_result = tflite_result.astype(np.single)
      self.assertEqual(iree_result.shape, tflite_result.shape)
      maxError = np.max(np.abs(iree_result - tflite_result))
      absl.logging.info("Max error (%d): %f", i, maxError)

  def setup_tflite(self):
    absl.logging.info("Setting up tflite interpreter")
    self.tflite_interpreter = tf.lite.Interpreter(model_path=self.tflite_file)
    self.tflite_interpreter.allocate_tensors()
    self.input_details = self.tflite_interpreter.get_input_details()
    self.output_details = self.tflite_interpreter.get_output_details()

  def setup_iree(self):
    absl.logging.info("Setting up iree runtime")
    with open(self.binary, 'rb') as f:
      config = iree_rt.Config(configs[absl.flags.FLAGS.config])
      self.iree_context = iree_rt.SystemContext(config=config)
      vm_module = iree_rt.VmModule.from_flatbuffer(f.read())
      self.iree_context.add_vm_module(vm_module)

  def invoke_tflite(self, args):
    for i, input in enumerate(args):
      self.tflite_interpreter.set_tensor(self.input_details[i]['index'], input)
    start = time.perf_counter()
    self.tflite_interpreter.invoke()
    end = time.perf_counter()
    tflite_results = []
    absl.logging.info(f"Invocation time: {end - start:0.4f} seconds")
    for output_detail in self.output_details:
      tflite_results.append(
          np.array(self.tflite_interpreter.get_tensor(output_detail['index'])))

    for i in range(len(self.output_details)):
      dtype = self.output_details[i]["dtype"]
      tflite_results[i] = tflite_results[i].astype(dtype)
    return tflite_results

  def invoke_iree(self, args):
    invoke = self.iree_context.modules.module["main"]
    start = time.perf_counter()
    iree_results = invoke(*args)
    end = time.perf_counter()
    absl.logging.info(f"Invocation time: {end - start:0.4f} seconds")
    if not isinstance(iree_results, tuple):
      iree_results = (iree_results,)
    return iree_results

  def compile_and_execute(self):
    self.assertIsNotNone(self.model_path)

    absl.logging.info("Setting up for IREE")
    iree_tflite_compile.compile_file(
        self.tflite_file,
        input_type="tosa",
        output_file=self.binary,
        save_temp_tfl_input=self.tflite_ir,
        save_temp_iree_input=self.iree_ir,
        target_backends=[targets[absl.flags.FLAGS.config]],
        import_only=False)

    self.setup_tflite()
    self.setup_iree()

    absl.logging.info("Setting up test inputs")
    args = self.generate_inputs(self.input_details)

    absl.logging.info("Invoking TFLite")
    tflite_results = self.invoke_tflite(args)

    absl.logging.info("Invoke IREE")
    iree_results = self.invoke_iree(args)

    # Fix type information for unsigned cases.
    iree_results = list(iree_results)
    for i in range(len(self.output_details)):
      dtype = self.output_details[i]["dtype"]
      iree_results[i] = iree_results[i].astype(dtype)

    self.compare_results(iree_results, tflite_results, self.output_details)
