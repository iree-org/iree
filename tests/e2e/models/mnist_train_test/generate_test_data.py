# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax
import numpy.random as npr
import datasets
import jax.core
import jax.numpy as jnp
from jax import grad, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.tree_util import tree_flatten
from iree.jax import (
    like,
    kernel,
    Program,
)
import numpy as np
import argparse


def get_example_batch():
  batch_size = 128
  train_images, train_labels, _, _ = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()
  return next(batches)


def get_model():
  init_random_params, predict = stax.serial(
      Dense(128),
      Relu,
      Dense(128),
      Relu,
      Dense(10),
      LogSoftmax,
  )
  return init_random_params, predict


def loss(params, batch, predict_fn):
  inputs, targets = batch
  preds = predict_fn(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))


def create_iree_jax_module():
  init_random_params, forward = get_model()

  rng = random.PRNGKey(12345)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, opt_get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  example_batch = get_example_batch()

  class IreeJaxMnistModule(Program):
    _opt_state = opt_state

    def get_params(self):
      return opt_get_params(self._opt_state)

    def get_opt_state(self):
      return self._opt_state

    def set_opt_state(self, new_opt_state=like(opt_state)):
      self._opt_state = new_opt_state

    def initialize(self, rng=like(rng)):
      self._opt_state = self._initialize_optimizer(rng)

    def update(self, batch=like(example_batch)):
      new_opt_state = self._update_step(batch, self._opt_state)
      self._opt_state = new_opt_state

    def forward(self, inputs=like(example_batch[0])):
      return self._forward(opt_get_params(self._opt_state), inputs)

    @kernel
    def _initialize_optimizer(rng):
      _, init_params = init_random_params(rng, (-1, 28 * 28))
      return opt_init(init_params)

    @kernel
    def _update_step(batch, opt_state):
      params = opt_get_params(opt_state)
      return opt_update(0, grad(loss)(params, batch, forward), opt_state)

    @kernel
    def _forward(params, inputs):
      return forward(params, inputs)

  return IreeJaxMnistModule()


def build_mlir_module(output_filepath):
  module = create_iree_jax_module()
  with open(output_filepath, "wb") as f:
    Program.get_mlir_module(module).operation.write_bytecode(f)


def build_jax_module():
  init_random_params, forward = get_model()

  rng = random.PRNGKey(12345)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, opt_get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  example_batch = get_example_batch()

  class JaxMnistModule:
    _opt_state = opt_state

    def get_params(self):
      return opt_get_params(self._opt_state)

    def get_opt_state(self):
      return self._opt_state

    def set_opt_state(self, new_opt_state):
      self._opt_state = new_opt_state

    def initialize(self, rng):
      self._opt_state = JaxMnistModule._initialize_optimizer(rng)

    def update(self, batch):
      new_opt_state = JaxMnistModule._update_step(batch, self._opt_state)
      self._opt_state = new_opt_state

    def forward(self, inputs):
      return JaxMnistModule._forward(opt_get_params(self._opt_state), inputs)

    @jax.jit
    def _initialize_optimizer(rng):
      _, init_params = init_random_params(rng, (-1, 28 * 28))
      return opt_init(init_params)

    @jax.jit
    def _update_step(batch, opt_state):
      params = opt_get_params(opt_state)
      return opt_update(0, grad(loss)(params, batch, forward), opt_state)

    @jax.jit
    def _forward(params, inputs):
      return forward(params, inputs)

  return JaxMnistModule()


def generate_test_data(output_mlir_filepath: str, batch_filepath: str,
                       expected_optimizer_state_after_init_filepath: str,
                       expected_optimizer_state_after_train_step_filepath: str,
                       expected_prediction_after_train_step_filepath: str):
  build_mlir_module(output_mlir_filepath)
  example_batch = get_example_batch()
  np.savez_compressed(batch_filepath, *example_batch)
  jax_module = build_jax_module()
  jax_module.update(example_batch)
  np.savez_compressed(expected_optimizer_state_after_train_step_filepath,
                      *tree_flatten(jax_module.get_opt_state())[0])
  prediction_jax = jax_module.forward(example_batch[0])
  np.savez_compressed(expected_prediction_after_train_step_filepath,
                      prediction_jax)
  rng = random.PRNGKey(6789)
  jax_module.initialize(rng)
  np.savez_compressed(expected_optimizer_state_after_init_filepath,
                      *tree_flatten(jax_module.get_opt_state())[0])


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_mlir_filepath",
                      help="Output to the compiled IREE Jax MLIR model.",
                      type=str,
                      default="mnist_train.mlirbc")
  parser.add_argument("--batch_filepath", type=str, default="batch.npz")
  parser.add_argument("--expected_optimizer_state_after_init_filepath",
                      type=str,
                      default="expected_optimizer_state_after_init.npz")
  parser.add_argument("--expected_optimizer_state_after_train_step_filepath",
                      type=str,
                      default="expected_optimizer_state_after_train_step.npz")
  parser.add_argument("--expected_prediction_after_train_step_filepath",
                      type=str,
                      default="expected_prediction_after_train_step.npz")
  return parser.parse_args()


def generate_test_data_cli():
  kwargs = vars(parse_args())
  generate_test_data(**kwargs)


if __name__ == "__main__":
  generate_test_data_cli()
