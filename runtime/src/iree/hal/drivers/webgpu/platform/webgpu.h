// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_PLATFORM_WEBGPU_H_
#define IREE_HAL_DRIVERS_WEBGPU_PLATFORM_WEBGPU_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#define WGPU_SKIP_PROCS 1
#if defined(IREE_PLATFORM_EMSCRIPTEN)
#include <emscripten/html5_webgpu.h>
#else
#include "third_party/webgpu-headers/webgpu.h"  // IWYU pragma: export
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// WebGPU API utilities
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
#define WGPU_DEBUG_LABEL(str) str
#else
#define WGPU_DEBUG_LABEL(str) NULL
#endif  // NDEBUG

//===----------------------------------------------------------------------===//
// Implementation compatibility layer
//===----------------------------------------------------------------------===//
// The webgpu-native headers don't yet line up across implementations or expose
// everything we need. These methods attempt to paper over that such that we
// can avoid including implementation-specific headers and #ifdefing everywhere.

// Methods for dropping references to objects.
// The base header does have some *Destroy methods but they are not implemented
// anywhere yet and the naming is incorrect as they are just dropping the user
// reference to the object (the implementation retains them as long as needed).
// Discussion here: https://github.com/webgpu-native/webgpu-headers/pull/15

void iree_wgpuBindGroupDrop(WGPUBindGroup bindGroup);
void iree_wgpuBindGroupLayoutDrop(WGPUBindGroupLayout bindGroupLayout);
void iree_wgpuBufferDrop(WGPUBuffer buffer);
void iree_wgpuCommandBufferDrop(WGPUCommandBuffer commandBuffer);
void iree_wgpuCommandEncoderDrop(WGPUCommandEncoder commandEncoder);
void iree_wgpuComputePipelineDrop(WGPUComputePipeline computePipeline);
void iree_wgpuPipelineLayoutDrop(WGPUPipelineLayout pipelineLayout);
void iree_wgpuQuerySetDrop(WGPUQuerySet querySet);
void iree_wgpuShaderModuleDrop(WGPUShaderModule shaderModule);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_PLATFORM_WEBGPU_H_
