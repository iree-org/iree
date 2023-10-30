## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from benchmark_suites.iree import benchmark_collections
from e2e_test_framework.definitions import common_definitions, iree_definitions

MODEL = common_definitions.Model(
    id="dummy-model-1234",
    name="dummy-model",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url="https://example.com/xyz.mlir",
    entry_function="main",
    input_types=["1xf32"],
)
IMPORTED_MODEL = iree_definitions.ImportedModel.from_model(MODEL)
COMPILE_CONFIG = iree_definitions.CompileConfig.build(
    id="dummy-config-1234",
    compile_targets=[
        iree_definitions.CompileTarget(
            target_architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_abi=iree_definitions.TargetABI.LINUX_GNU,
        )
    ],
)
EXEC_CONFIG = iree_definitions.ModuleExecutionConfig.build(
    id="dummy-exec-1234",
    loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
    driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
)
DEVICE_SPEC = common_definitions.DeviceSpec.build(
    id="dummy-device-1234",
    device_name="dummy-device",
    architecture=common_definitions.DeviceArchitecture.CUDA_SM80,
    host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
)


class BenchmarkCollectionsTest(unittest.TestCase):
    def test_validate_gen_configs(self):
        config_a = iree_definitions.ModuleGenerationConfig(
            composite_id="a",
            name="model-name (A.RCH)[tag_0,tag_1]",
            tags=[],
            presets=["default"],
            imported_model=IMPORTED_MODEL,
            compile_config=COMPILE_CONFIG,
            compile_flags=[],
        )
        config_b = iree_definitions.ModuleGenerationConfig(
            composite_id="b",
            name="name-b",
            tags=[],
            presets=["default"],
            imported_model=IMPORTED_MODEL,
            compile_config=COMPILE_CONFIG,
            compile_flags=[],
        )

        benchmark_collections.validate_gen_configs([config_a, config_b])

    def test_validate_gen_configs_duplicate_name(self):
        config_a = iree_definitions.ModuleGenerationConfig(
            composite_id="a",
            name="name",
            tags=[],
            presets=["default"],
            imported_model=IMPORTED_MODEL,
            compile_config=COMPILE_CONFIG,
            compile_flags=[],
        )
        config_b = iree_definitions.ModuleGenerationConfig(
            composite_id="b",
            name="name",
            tags=[],
            presets=["default"],
            imported_model=IMPORTED_MODEL,
            compile_config=COMPILE_CONFIG,
            compile_flags=[],
        )

        self.assertRaises(
            ValueError,
            lambda: benchmark_collections.validate_gen_configs([config_a, config_b]),
        )

    def test_validate_gen_configs_disallowed_characters(self):
        config_a = iree_definitions.ModuleGenerationConfig(
            composite_id="a",
            name="name+a",
            tags=[],
            presets=["default"],
            imported_model=IMPORTED_MODEL,
            compile_config=COMPILE_CONFIG,
            compile_flags=[],
        )

        self.assertRaises(
            ValueError,
            lambda: benchmark_collections.validate_gen_configs([config_a]),
        )

    def test_validate_gen_configs_duplicate_id(self):
        config_a = iree_definitions.ModuleGenerationConfig(
            composite_id="x",
            name="name-a",
            tags=[],
            presets=["default"],
            imported_model=IMPORTED_MODEL,
            compile_config=COMPILE_CONFIG,
            compile_flags=[],
        )
        config_b = iree_definitions.ModuleGenerationConfig(
            composite_id="x",
            name="name-b",
            tags=[],
            presets=["default"],
            imported_model=IMPORTED_MODEL,
            compile_config=COMPILE_CONFIG,
            compile_flags=[],
        )

        self.assertRaises(
            ValueError,
            lambda: benchmark_collections.validate_gen_configs([config_a, config_b]),
        )

    def test_validate_run_configs(self):
        config_a = iree_definitions.E2EModelRunConfig(
            composite_id="a",
            name="model-name (A.RCH)[tag_0,tag_1] @ device",
            tags=[],
            presets=["default"],
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=IMPORTED_MODEL,
                compile_config=COMPILE_CONFIG,
            ),
            module_execution_config=EXEC_CONFIG,
            target_device_spec=DEVICE_SPEC,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
            run_flags=[],
        )
        config_b = iree_definitions.E2EModelRunConfig(
            composite_id="b",
            name="name-b",
            tags=[],
            presets=["default"],
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=IMPORTED_MODEL,
                compile_config=COMPILE_CONFIG,
            ),
            module_execution_config=EXEC_CONFIG,
            target_device_spec=DEVICE_SPEC,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
            run_flags=[],
        )

        benchmark_collections.validate_run_configs([config_a, config_b])

    def test_validate_run_configs_duplicate_name(self):
        config_a = iree_definitions.E2EModelRunConfig(
            composite_id="a",
            name="name",
            tags=[],
            presets=["default"],
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=IMPORTED_MODEL,
                compile_config=COMPILE_CONFIG,
            ),
            module_execution_config=EXEC_CONFIG,
            target_device_spec=DEVICE_SPEC,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
            run_flags=[],
        )
        config_b = iree_definitions.E2EModelRunConfig(
            composite_id="b",
            name="name",
            tags=[],
            presets=["default"],
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=IMPORTED_MODEL,
                compile_config=COMPILE_CONFIG,
            ),
            module_execution_config=EXEC_CONFIG,
            target_device_spec=DEVICE_SPEC,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
            run_flags=[],
        )

        self.assertRaises(
            ValueError,
            lambda: benchmark_collections.validate_run_configs([config_a, config_b]),
        )

    def test_validate_run_configs_duplicate_id(self):
        config_a = iree_definitions.E2EModelRunConfig(
            composite_id="x",
            name="name-a",
            tags=[],
            presets=["default"],
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=IMPORTED_MODEL,
                compile_config=COMPILE_CONFIG,
            ),
            module_execution_config=EXEC_CONFIG,
            target_device_spec=DEVICE_SPEC,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
            run_flags=[],
        )
        config_b = iree_definitions.E2EModelRunConfig(
            composite_id="x",
            name="name-b",
            tags=[],
            presets=["default"],
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=IMPORTED_MODEL,
                compile_config=COMPILE_CONFIG,
            ),
            module_execution_config=EXEC_CONFIG,
            target_device_spec=DEVICE_SPEC,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
            run_flags=[],
        )

        self.assertRaises(
            ValueError,
            lambda: benchmark_collections.validate_run_configs([config_a, config_b]),
        )

    def test_validate_run_configs_disallowed_characters(self):
        config = iree_definitions.E2EModelRunConfig(
            composite_id="x",
            name="name+a",
            tags=[],
            presets=["default"],
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=IMPORTED_MODEL,
                compile_config=COMPILE_CONFIG,
            ),
            module_execution_config=EXEC_CONFIG,
            target_device_spec=DEVICE_SPEC,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
            run_flags=[],
        )

        self.assertRaises(
            ValueError,
            lambda: benchmark_collections.validate_run_configs([config]),
        )


if __name__ == "__main__":
    unittest.main()
