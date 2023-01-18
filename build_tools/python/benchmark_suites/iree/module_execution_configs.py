## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines ModuleExecutionConfig for benchmarks."""

from e2e_test_framework.definitions import iree_definitions
from e2e_test_framework import unique_ids

ELF_LOCAL_SYNC_CONFIG = iree_definitions.ModuleExecutionConfig(
    id=unique_ids.IREE_MODULE_EXECUTION_CONFIG_LOCAL_SYNC,
    tags=["full-inference", "default-flags"],
    loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
    driver=iree_definitions.RuntimeDriver.LOCAL_SYNC)

CUDA_CONFIG = iree_definitions.ModuleExecutionConfig(
    id=unique_ids.IREE_MODULE_EXECUTION_CONFIG_CUDA,
    tags=["full-inference", "default-flags"],
    loader=iree_definitions.RuntimeLoader.NONE,
    driver=iree_definitions.RuntimeDriver.CUDA)

VULKAN_CONFIG = iree_definitions.ModuleExecutionConfig(
    id=unique_ids.IREE_MODULE_EXECUTION_CONFIG_VULKAN,
    tags=["full-inference", "default-flags"],
    loader=iree_definitions.RuntimeLoader.NONE,
    driver=iree_definitions.RuntimeDriver.VULKAN)

VULKAN_BATCH_SIZE_16_CONFIG = iree_definitions.ModuleExecutionConfig(
    id=unique_ids.IREE_MODULE_EXECUTION_CONFIG_VULKAN_BATCH_SIZE_16,
    tags=["full-inference", "experimental-flags"],
    loader=iree_definitions.RuntimeLoader.NONE,
    driver=iree_definitions.RuntimeDriver.VULKAN,
    extra_flags=["--batch_size=16"])

VULKAN_BATCH_SIZE_32_CONFIG = iree_definitions.ModuleExecutionConfig(
    id=unique_ids.IREE_MODULE_EXECUTION_CONFIG_VULKAN_BATCH_SIZE_32,
    tags=["full-inference", "experimental-flags"],
    loader=iree_definitions.RuntimeLoader.NONE,
    driver=iree_definitions.RuntimeDriver.VULKAN,
    extra_flags=["--batch_size=32"])


def get_elf_local_task_config(thread_num: int):
  config_id = f"{unique_ids.IREE_MODULE_EXECUTION_CONFIG_LOCAL_TASK_BASE}-{thread_num}"
  return iree_definitions.ModuleExecutionConfig(
      id=config_id,
      tags=[f"{thread_num}-thread", "full-inference", "default-flags"],
      loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
      driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
      extra_flags=[f"--task_topology_group_count={thread_num}"])


def get_vmvx_local_task_config(thread_num: int):
  config_id = f"{unique_ids.IREE_MODULE_EXECUTION_CONFIG_VMVX_LOCAL_TASK_BASE}-{thread_num}"
  return iree_definitions.ModuleExecutionConfig(
      id=config_id,
      tags=[f"{thread_num}-thread", "full-inference", "default-flags"],
      loader=iree_definitions.RuntimeLoader.VMVX_MODULE,
      driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
      extra_flags=[f"--task_topology_group_count={thread_num}"])
