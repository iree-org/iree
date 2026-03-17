// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebGPU device setup for the simple embedding sample.
//
// Uses the pre-configured device handle provided by the JS host via the
// get_preconfigured_device bridge import. The JS entry point creates a
// GPUDevice via dawn (Node.js) or navigator.gpu (browser) before wasm starts.

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/api.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"

// Compiled module embedded here to avoid file IO:
#include "samples/simple_embedding/simple_embedding_test_bytecode_module_webgpu_c.h"

iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device) {
  // The JS host pre-configured a GPUDevice and stored the bridge handle via
  // context.preConfiguredDevice. The wrap API creates a HAL device from it
  // with an inline-mode proactor (no threads needed).
  iree_hal_webgpu_handle_t device_handle =
      iree_hal_webgpu_import_get_preconfigured_device();
  if (device_handle == 0) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "no pre-configured WebGPU device; ensure the JS host sets "
        "context.preConfiguredDevice before starting wasm");
  }
  return iree_hal_webgpu_device_wrap(iree_make_cstring_view("webgpu"),
                                     device_handle, host_allocator, out_device);
}

const iree_const_byte_span_t load_bytecode_module_data() {
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_simple_embedding_test_module_webgpu_create();
  return iree_make_const_byte_span(module_file_toc->data,
                                   module_file_toc->size);
}
