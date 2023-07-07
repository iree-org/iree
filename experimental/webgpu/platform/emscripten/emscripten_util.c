// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/platform/webgpu.h"

//===----------------------------------------------------------------------===//
// Implementation compatibility layer
//===----------------------------------------------------------------------===//

#define WGPU_EMSCRIPTEN_INSTANCE ((WGPUInstance)((uintptr_t)0xABADD00Du))

WGPUInstance wgpuCreateInstance(WGPUInstanceDescriptor const* descriptor) {
  // Emscripten does not have instances (yet?)
  // We use a sentinel value here so that we can do null checks in places for
  // implementations that do use instances.
  return WGPU_EMSCRIPTEN_INSTANCE;
}

void iree_wgpuBindGroupDrop(WGPUBindGroup bindGroup) {
  // Not implemented on the web / Emscripten.
}

void iree_wgpuBindGroupLayoutDrop(WGPUBindGroupLayout bindGroupLayout) {
  // Not implemented on the web / Emscripten.
}

void iree_wgpuBufferDrop(WGPUBuffer buffer) { wgpuBufferDestroy(buffer); }

void iree_wgpuCommandBufferDrop(WGPUCommandBuffer commandBuffer) {
  // Not implemented on the web / Emscripten.
}

void iree_wgpuCommandEncoderDrop(WGPUCommandEncoder commandEncoder) {
  // Not implemented on the web / Emscripten.
}

void iree_wgpuComputePipelineDrop(WGPUComputePipeline computePipeline) {
  // Not implemented on the web / Emscripten.
}

void iree_wgpuPipelineLayoutDrop(WGPUPipelineLayout pipelineLayout) {
  // Not implemented on the web / Emscripten.
}

void iree_wgpuQuerySetDrop(WGPUQuerySet querySet) {
  wgpuQuerySetDestroy(querySet);
}

void iree_wgpuShaderModuleDrop(WGPUShaderModule shaderModule) {
  // Not implemented on the web / Emscripten.
}
