## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines configs to help testing."""

from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_artifacts import model_artifacts

TFLITE_MODEL = common_definitions.Model(
    id="1234",
    name="tflite_m",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    source_url="https://example.com/xyz.tflite",
    entry_function="main",
    input_types=["1xf32"])
TF_MODEL = common_definitions.Model(
    id="5678",
    name="tf_m",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF,
    source_url="https://example.com/xyz_saved_model",
    entry_function="predict",
    input_types=["2xf32"])
LINALG_MODEL = common_definitions.Model(
    id="9012",
    name="linalg_m",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://example.com/xyz.mlir",
    entry_function="main",
    input_types=["3xf32"])

TFLITE_IMPORTED_MODEL = iree_definitions.ImportedModel(
    model=TFLITE_MODEL, dialect_type=iree_definitions.MLIRDialectType.TOSA)
TF_IMPORTED_MODEL = iree_definitions.ImportedModel(
    model=TF_MODEL, dialect_type=iree_definitions.MLIRDialectType.MHLO)
LINALG_IMPORTED_MODEL = iree_definitions.ImportedModel(
    model=LINALG_MODEL, dialect_type=iree_definitions.MLIRDialectType.LINALG)

COMPILE_CONFIG_A = iree_definitions.CompileConfig(
    id="config_a",
    tags=["defaults"],
    compile_targets=[
        iree_definitions.CompileTarget(
            target_architecture=common_definitions.DeviceArchitecture.
            X86_64_CASCADELAKE,
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_abi=iree_definitions.TargetABI.LINUX_GNU)
    ])
COMPILE_CONFIG_B = iree_definitions.CompileConfig(
    id="config_b",
    tags=["experimentals"],
    compile_targets=[
        iree_definitions.CompileTarget(
            target_architecture=common_definitions.DeviceArchitecture.
            ARMV8_2_A_GENERIC,
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_abi=iree_definitions.TargetABI.LINUX_ANDROID29)
    ])
