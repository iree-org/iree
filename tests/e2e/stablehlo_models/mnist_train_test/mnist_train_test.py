# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from typing import List, TypeVar
from urllib.request import urlretrieve

import numpy as np

from iree.compiler.tools import InputType, compile_file
from iree.runtime import load_vm_flatbuffer_file

MODEL_ARTIFACTS_URL = "https://storage.googleapis.com/iree-model-artifacts/mnist_train.2bec0cb356ae7c059e04624a627eb3b15b0a556cbd781bbed9f8d32e80a4311d.tar"

Tensor = TypeVar("Tensor")


def build_module(artifacts_dir: str):
    vmfb_file = os.path.join(artifacts_dir, "mnist_train.vmfb")
    compile_file(
        input_file=os.path.join(artifacts_dir, "mnist_train.mlirbc"),
        output_file=vmfb_file,
        target_backends=[args.target_backend],
        input_type=InputType.STABLEHLO,
    )
    # On Windows, the flatbuffer is mmap'd and cannot be deleted while in use.
    # So copy it to a temporary location and mmap from there (preserving the
    # artifacts as-is).
    with tempfile.NamedTemporaryFile(delete=False) as mmap_vmfb:
        mmap_vmfb.close()
        shutil.copyfile(vmfb_file, mmap_vmfb.name)

        def cleanup():
            os.unlink(mmap_vmfb.name)

        return load_vm_flatbuffer_file(
            mmap_vmfb.name, driver=args.driver, destroy_callback=cleanup
        )


def load_data(data_dir: str):
    batch = list(np.load(os.path.join(data_dir, "batch.npz")).values())
    expected_optimizer_state_after_init = list(
        np.load(
            os.path.join(data_dir, "expected_optimizer_state_after_init.npz")
        ).values()
    )
    expected_optimizer_state_after_train_step = list(
        np.load(
            os.path.join(data_dir, "expected_optimizer_state_after_train_step.npz")
        ).values()
    )
    expected_prediction_after_train_step = list(
        np.load(
            os.path.join(data_dir, "expected_prediction_after_train_step.npz")
        ).values()
    )[0]
    return (
        batch,
        expected_optimizer_state_after_init,
        expected_optimizer_state_after_train_step,
        expected_prediction_after_train_step,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_backend", type=str, default="llvm-cpu")
    parser.add_argument("--driver", type=str, default="local-task")
    return parser.parse_known_args()


DEFAULT_REL_TOLERANCE = 1e-5
DEFAULT_ABS_TOLERANCE = 1e-5


def allclose(
    a: Tensor, b: Tensor, rtol=DEFAULT_REL_TOLERANCE, atol=DEFAULT_ABS_TOLERANCE
):
    return np.allclose(np.asarray(a), np.asarray(b), rtol, atol)


def assert_array_list_compare(array_compare_fn, a: Tensor, b: Tensor):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_compare(array_compare_fn, x, y)


def assert_array_list_allclose(
    a: List[Tensor],
    b: List[Tensor],
    rtol=DEFAULT_REL_TOLERANCE,
    atol=DEFAULT_ABS_TOLERANCE,
):
    assert_array_list_compare(lambda x, y: allclose(x, y, rtol, atol), a, b)


def download_test_data(out_path: str):
    urlretrieve(MODEL_ARTIFACTS_URL, out_path)


def extract_test_data(archive_path: str, out_dir: str):
    with tarfile.open(archive_path) as tar:
        tar.extractall(out_dir)


class MnistTrainTest(unittest.TestCase):
    def test_mnist_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = os.path.join(tmp_dir, "mnist_train.tar")
            download_test_data(archive_path)
            extract_test_data(archive_path, tmp_dir)
            module = build_module(tmp_dir)
            (
                batch,
                expected_optimizer_state_after_init,
                expected_optimizer_state_after_train_step,
                expected_prediction_after_train_step,
            ) = load_data(tmp_dir)

        module.update(*batch)
        assert_array_list_allclose(
            module.get_opt_state(), expected_optimizer_state_after_train_step
        )
        prediction = module.forward(batch[0])
        np.testing.assert_allclose(
            prediction,
            expected_prediction_after_train_step,
            DEFAULT_REL_TOLERANCE,
            DEFAULT_ABS_TOLERANCE,
        )
        rng_state = np.array([0, 6789], dtype=np.int32)
        module.initialize(rng_state)
        assert_array_list_allclose(
            module.get_opt_state(), expected_optimizer_state_after_init
        )


if __name__ == "__main__":
    args, remaining_args = parse_args()
    unittest.main(argv=[sys.argv[0]] + remaining_args)
