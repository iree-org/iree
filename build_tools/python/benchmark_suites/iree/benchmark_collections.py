## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates all benchmarks."""

import collections
import re
from typing import List, Tuple, Sequence

from e2e_test_artifacts import iree_artifacts
from e2e_test_framework.definitions import iree_definitions
from benchmark_suites.iree import (
    arm64_benchmarks,
    benchmark_presets,
    riscv_benchmarks,
    x86_64_benchmarks,
    adreno_benchmarks,
    cuda_benchmarks,
    mali_benchmarks,
    vulkan_nvidia_benchmarks,
    vmvx_benchmarks,
)

COMPILE_STATS_ID_SUFFIX = "-compile-stats"
ALLOWED_NAME_FORMAT = re.compile(r"^[0-9a-zA-Z.,\-_()\[\] @]+$")


def validate_gen_configs(
    gen_configs: Sequence[iree_definitions.ModuleGenerationConfig],
):
    """Check the uniqueness and name format of module generation configs."""

    ids_to_configs = {}
    names_to_configs = {}
    for gen_config in gen_configs:
        if not ALLOWED_NAME_FORMAT.match(gen_config.name):
            raise ValueError(
                f"Module generation config name: '{gen_config.name}' doesn't"
                f" follow the format '{ALLOWED_NAME_FORMAT.pattern}'"
            )

        if gen_config.composite_id in ids_to_configs:
            raise ValueError(
                "Two module generation configs have the same ID:\n\n"
                f"{repr(gen_config)}\n\n"
                f"{repr(ids_to_configs[gen_config.composite_id])}"
            )
        ids_to_configs[gen_config.composite_id] = gen_config

        if gen_config.name in names_to_configs:
            raise ValueError(
                "Two module generation configs have the same name:\n\n"
                f"{repr(gen_config)}\n\n"
                f"{repr(names_to_configs[gen_config.name])}"
            )
        names_to_configs[gen_config.name] = gen_config


def validate_run_configs(
    run_configs: Sequence[iree_definitions.E2EModelRunConfig],
):
    """Check the uniqueness and name format of E2E model run configs."""

    ids_to_configs = {}
    names_to_configs = {}
    for run_config in run_configs:
        if not ALLOWED_NAME_FORMAT.match(run_config.name):
            raise ValueError(
                f"E2E model run config name: '{run_config.name}' doesn't"
                f" follow the format '{ALLOWED_NAME_FORMAT.pattern}'"
            )

        if run_config.composite_id in ids_to_configs:
            raise ValueError(
                "Two e2e model run configs have the same ID:\n\n"
                f"{repr(run_config)}\n\n"
                f"{repr(ids_to_configs[run_config.composite_id])}"
            )
        ids_to_configs[run_config.composite_id] = run_config

        if run_config.name in names_to_configs:
            raise ValueError(
                "Two e2e model run configs have the same name:\n\n"
                f"{repr(run_config)}\n\n"
                f"{repr(names_to_configs[run_config.name])}"
            )
        names_to_configs[run_config.name] = run_config


def generate_benchmarks() -> (
    Tuple[
        List[iree_definitions.ModuleGenerationConfig],
        List[iree_definitions.E2EModelRunConfig],
    ]
):
    """Generate the benchmark suite."""

    benchmarks = [
        x86_64_benchmarks.Linux_x86_64_Benchmarks(),
        cuda_benchmarks.Linux_CUDA_Benchmarks(),
        riscv_benchmarks.Linux_RV64_Benchmarks(),
        riscv_benchmarks.Linux_RV32_Benchmarks(),
        arm64_benchmarks.Android_ARM64_Benchmarks(),
        adreno_benchmarks.Android_Adreno_Benchmarks(),
        mali_benchmarks.Android_Mali_Benchmarks(),
        vulkan_nvidia_benchmarks.Linux_Vulkan_NVIDIA_Benchmarks(),
        vmvx_benchmarks.VMVX_Benchmarks(),
    ]
    all_run_configs: List[iree_definitions.E2EModelRunConfig] = []
    for benchmark in benchmarks:
        run_configs = benchmark.generate()
        all_run_configs += run_configs

    # Collect all module generation configs in run configs.
    all_gen_configs = {}
    gen_config_dependents = collections.defaultdict(list)
    for run_config in all_run_configs:
        gen_config = run_config.module_generation_config
        all_gen_configs[gen_config.composite_id] = gen_config
        gen_config_dependents[gen_config.composite_id].append(run_config)

    all_gen_configs = list(all_gen_configs.values())

    validate_gen_configs(all_gen_configs)
    validate_run_configs(all_run_configs)

    compile_stats_gen_configs = []
    # For now we simply track compilation statistics of all modules.
    for gen_config in all_gen_configs:
        compile_config = gen_config.compile_config
        # Use POSIX path, see the comment of iree_definitions.MODULE_DIR_VARIABLE.
        scheduling_stats_path = f"{iree_definitions.MODULE_DIR_VARIABLE}/{iree_artifacts.SCHEDULING_STATS_FILENAME}"
        compile_stats_config = iree_definitions.CompileConfig.build(
            id=compile_config.id + COMPILE_STATS_ID_SUFFIX,
            tags=compile_config.tags + ["compile-stats"],
            compile_targets=compile_config.compile_targets,
            extra_flags=compile_config.extra_flags
            + [
                # Enable zip polyglot to provide component sizes.
                "--iree-vm-emit-polyglot-zip=true",
                # Disable debug symbols to provide correct component sizes.
                "--iree-llvmcpu-debug-symbols=false",
                # Dump scheduling statistics
                "--iree-scheduling-dump-statistics-format=json",
                f"--iree-scheduling-dump-statistics-file={scheduling_stats_path}",
            ],
        )

        dependents = gen_config_dependents[gen_config.composite_id]
        compile_stats_presets = set()
        for dependent in dependents:
            dep_presets = set(dependent.presets)
            # Assign compilation benchmark presets based on the size of the
            # original benchmarks.
            # A benchmark can be in default and large presets at the same time,
            # for example, batch-1 benchmark is added to default for sanity
            # check. So check both cases.
            if dep_presets.intersection(benchmark_presets.DEFAULT_PRESETS):
                compile_stats_presets.add(benchmark_presets.COMP_STATS)
            if dep_presets.intersection(benchmark_presets.LARGE_PRESETS):
                compile_stats_presets.add(benchmark_presets.COMP_STATS_LARGE)

        compile_stats_gen_configs.append(
            iree_definitions.ModuleGenerationConfig.build(
                imported_model=gen_config.imported_model,
                compile_config=compile_stats_config,
                presets=sorted(compile_stats_presets),
                tags=gen_config.tags,
            )
        )
    all_gen_configs += compile_stats_gen_configs

    return (all_gen_configs, all_run_configs)
