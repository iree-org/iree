## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE Mali GPU benchmarks."""

from typing import List, Sequence, Tuple
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.models import tflite_models
from e2e_test_framework.device_specs import device_collections
from benchmark_suites.iree import module_execution_configs
import benchmark_suites.iree.utils


class Android_Mali_Benchmarks(object):
  """Benchmarks on Android devices with Mali GPU."""

  VALHALL_MALI_GPU_TARGET = iree_definitions.CompileTarget(
      target_backend=iree_definitions.TargetBackend.VULKAN_SPIRV,
      target_architecture=common_definitions.DeviceArchitecture.VALHALL_MALI,
      target_abi=iree_definitions.TargetABI.VULKAN_ANDROID31)
  DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_VALHALL_MALI_DEFAULTS,
      tags=["default-flags"],
      compile_targets=[VALHALL_MALI_GPU_TARGET])
  FUSE_PADDING_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_VALHALL_MALI_FUSE_PADDING,
      tags=["experimental-flags", "fuse-padding"],
      compile_targets=[VALHALL_MALI_GPU_TARGET],
      extra_flags=["--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"])
  # Kernel execution
  # Note that for kernel-execution benchmarks batch_size/repeat-count need to be
  # low enough that the whole dispatch completes within an OS-specific timeout.
  # Otherwise you'll get error like:
  # ```
  # INTERNAL; VK_ERROR_DEVICE_LOST; vkQueueSubmit; while invoking native function
  # hal.ex.submit_and_wait; while calling import;
  # ```
  FUSE_PADDING_REPEATED_KERNEL_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.
      IREE_COMPILE_CONFIG_ANDROID_VALHALL_MALI_FUSE_PADDING_REPEATED_KERNEL,
      tags=["experimental-flags", "fuse-padding", "repeated-kernel"],
      compile_targets=[VALHALL_MALI_GPU_TARGET],
      extra_flags=FUSE_PADDING_COMPILE_CONFIG.extra_flags +
      ["--iree-hal-benchmark-dispatch-repeat-count=32"])
  FUSE_PADDING_REPEATED_KERNEL_RUN_FLAGS = ["--batch_size=32"]

  FP32_MODELS = [
      tflite_models.DEEPLABV3_FP32,
      tflite_models.MOBILESSD_FP32,
      tflite_models.POSENET_FP32,
      tflite_models.MOBILEBERT_FP32,
      tflite_models.MOBILENET_V2,
      tflite_models.MOBILENET_V3SMALL,
  ]
  FP16_MODELS = [tflite_models.MOBILEBERT_FP16]
  QUANT_MODELS = [
      tflite_models.MOBILEBERT_INT8,
      tflite_models.EFFICIENTNET_INT8,
      tflite_models.PERSON_DETECT_INT8,
  ]

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    default_gen_configs = self._get_module_generation_configs(
        compile_config=self.DEFAULT_COMPILE_CONFIG,
        fp32_models=self.FP32_MODELS,
        fp16_models=self.FP16_MODELS,
        quant_models=self.QUANT_MODELS)
    fuse_padding_gen_configs = self._get_module_generation_configs(
        compile_config=self.FUSE_PADDING_COMPILE_CONFIG,
        fp32_models=self.FP32_MODELS,
        fp16_models=self.FP16_MODELS,
        quant_models=self.QUANT_MODELS)
    fuse_padding_repeated_kernel_gen_configs = self._get_module_generation_configs(
        compile_config=self.FUSE_PADDING_REPEATED_KERNEL_COMPILE_CONFIG,
        fp32_models=self.FP32_MODELS,
        fp16_models=self.FP16_MODELS,
        quant_models=self.QUANT_MODELS)

    mali_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.VALHALL_MALI,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A)
    run_configs = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=default_gen_configs +
        fuse_padding_gen_configs,
        module_execution_configs=[module_execution_configs.VULKAN_CONFIG],
        device_specs=mali_devices)
    run_configs += benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=fuse_padding_repeated_kernel_gen_configs,
        module_execution_configs=[
            module_execution_configs.VULKAN_BATCH_SIZE_32_CONFIG
        ],
        device_specs=mali_devices)

    gen_configs = (default_gen_configs + fuse_padding_gen_configs +
                   fuse_padding_repeated_kernel_gen_configs)
    return (gen_configs, run_configs)

  def _get_module_generation_configs(
      self, compile_config: iree_definitions.CompileConfig,
      fp32_models: Sequence[common_definitions.Model],
      fp16_models: Sequence[common_definitions.Model],
      quant_models: Sequence[common_definitions.Model]
  ) -> List[iree_definitions.ModuleGenerationConfig]:
    demote_compile_config = iree_definitions.CompileConfig.build(
        id=compile_config.id + "-demote-f32-to-16",
        tags=compile_config.tags + ["demote-f32-to-f16"],
        compile_targets=compile_config.compile_targets,
        extra_flags=compile_config.extra_flags +
        ["--iree-flow-demote-f32-to-f16"])
    return [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=compile_config,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in fp32_models
    ] + [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=demote_compile_config,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in fp16_models
    ] + [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=compile_config,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in quant_models
    ]
