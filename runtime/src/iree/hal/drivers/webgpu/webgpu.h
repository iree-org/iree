// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebGPU spec constants and bridge wire-format encodings.
//
// Numeric flag values (GPUBufferUsage, GPUShaderStage, GPUMapMode) are taken
// directly from the WebGPU specification. Bridge-specific encodings (buffer
// binding type) map WebGPU string enums to integer values for the wasm↔JS
// boundary — these must match the corresponding decode tables in
// webgpu_imports.mjs.

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// GPUBufferUsage (spec flags)
//===----------------------------------------------------------------------===//

enum iree_hal_webgpu_buffer_usage_bits_t {
  IREE_HAL_WEBGPU_BUFFER_USAGE_MAP_READ = 0x0001,
  IREE_HAL_WEBGPU_BUFFER_USAGE_MAP_WRITE = 0x0002,
  IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_SRC = 0x0004,
  IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_DST = 0x0008,
  IREE_HAL_WEBGPU_BUFFER_USAGE_INDEX = 0x0010,
  IREE_HAL_WEBGPU_BUFFER_USAGE_VERTEX = 0x0020,
  IREE_HAL_WEBGPU_BUFFER_USAGE_UNIFORM = 0x0040,
  IREE_HAL_WEBGPU_BUFFER_USAGE_STORAGE = 0x0080,
  IREE_HAL_WEBGPU_BUFFER_USAGE_INDIRECT = 0x0100,
  IREE_HAL_WEBGPU_BUFFER_USAGE_QUERY_RESOLVE = 0x0200,
};
typedef uint32_t iree_hal_webgpu_buffer_usage_t;

//===----------------------------------------------------------------------===//
// GPUShaderStage (spec flags)
//===----------------------------------------------------------------------===//

enum iree_hal_webgpu_shader_stage_bits_t {
  IREE_HAL_WEBGPU_SHADER_STAGE_VERTEX = 0x1,
  IREE_HAL_WEBGPU_SHADER_STAGE_FRAGMENT = 0x2,
  IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE = 0x4,
};
typedef uint32_t iree_hal_webgpu_shader_stage_t;

//===----------------------------------------------------------------------===//
// GPUMapMode (spec flags)
//===----------------------------------------------------------------------===//

enum iree_hal_webgpu_map_mode_bits_t {
  IREE_HAL_WEBGPU_MAP_MODE_READ = 0x1,
  IREE_HAL_WEBGPU_MAP_MODE_WRITE = 0x2,
};
typedef uint32_t iree_hal_webgpu_map_mode_t;

//===----------------------------------------------------------------------===//
// GPUBufferBindingType (bridge wire encoding)
//===----------------------------------------------------------------------===//

// Integer encoding of GPUBufferBindingType for the wasm↔JS boundary.
// The JS side decodes via: ['uniform', 'storage', 'read-only-storage'][value].
enum iree_hal_webgpu_buffer_binding_type_e {
  IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_UNIFORM = 0,
  IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_STORAGE = 1,
  IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE = 2,
};
typedef uint32_t iree_hal_webgpu_buffer_binding_type_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_H_
