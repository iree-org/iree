// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/platform/webgpu.h"

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

//===----------------------------------------------------------------------===//
// Speculative WebGPU API additions
//===----------------------------------------------------------------------===//

static void iree_hal_webgpu_buffer_map_sync_callback(
    WGPUBufferMapAsyncStatus status, void* userdata) {
  IREEWGPUBufferMapSyncStatus* sync_status =
      (IREEWGPUBufferMapSyncStatus*)userdata;
  switch (status) {
    case WGPUBufferMapAsyncStatus_Success:
      *sync_status = IREEWGPUBufferMapSyncStatus_Success;
      break;
    case WGPUBufferMapAsyncStatus_Error:
      *sync_status = IREEWGPUBufferMapSyncStatus_Error;
      break;
    default:
    case WGPUBufferMapAsyncStatus_Unknown:
      *sync_status = IREEWGPUBufferMapSyncStatus_Unknown;
      break;
    case WGPUBufferMapAsyncStatus_DeviceLost:
      *sync_status = IREEWGPUBufferMapSyncStatus_DeviceLost;
      break;
  }
}

IREEWGPUBufferMapSyncStatus iree_wgpuBufferMapSync(WGPUDevice device,
                                                   WGPUBuffer buffer,
                                                   WGPUMapModeFlags mode,
                                                   size_t offset, size_t size) {
  IREEWGPUBufferMapSyncStatus status = IREEWGPUBufferMapSyncStatus_Unknown;
  wgpuBufferMapAsync(buffer, mode, offset, size,
                     iree_hal_webgpu_buffer_map_sync_callback, &status);
  // TODO(scotttodd): poll / wait somehow, or implement sync mapping differently
  // wgpuDevicePoll(device, /*force_wait=*/true);
  return status;
}
