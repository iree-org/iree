# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class VmfbManager:
    sdxl_clip_cpu_vmfb = None
    sdxl_vae_cpu_vmfb = None
    sdxl_unet_fp16_cpu_vmfb = None
    sdxl_clip_rocm_vmfb = None
    sdxl_vae_rocm_vmfb = None
    sdxl_unet_fp16_rocm_vmfb = None
    sdxl_punet_int8_fp16_rocm_vmfb = None
    sdxl_punet_int8_fp8_rocm_vmfb = None
    sdxl_unet_fp16_cpu_pipeline_vmfb = None
    sdxl_unet_fp16_rocm_pipeline_vmfb = None
    sd3_clip_cpu_vmfb = None
    sd3_vae_cpu_vmfb = None
    sd3_mmdit_cpu_vmfb = None
    sd3_clip_rocm_vmfb = None
    sd3_vae_rocm_vmfb = None
    sd3_mmdit_rocm_vmfb = None
