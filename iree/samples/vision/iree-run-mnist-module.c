// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This sample uses iree/tools/utils/image_util to load a hand-written image
// as an iree_hal_buffer_view_t then passes it to the bytecode module built
// from mnist.mlir on the dylib-llvm-aot backend.

#include <float.h>

#include "iree/runtime/api.h"
#include "iree/samples/vision/mnist_bytecode_module_c.h"
#include "iree/tools/utils/image_util.h"

iree_status_t Run(const iree_string_view_t image_path) {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));

  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("dylib"), &device));

  // Create one session per loaded module to hold the module state.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  const struct iree_file_toc_t* module_file = mnist_bytecode_module_c_create();

  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_memory(
      session, iree_make_const_byte_span(module_file->data, module_file->size),
      iree_allocator_null()));

  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.predict"), &call));

  // Prepare the input hal buffer view with image_util library.
  // The input of the mmist model is single 28x28 pixel image as a
  // tensor<1x28x28x1xf32>, with pixels in [0.0, 1.0].
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_hal_dim_t buffer_shape[] = {1, 28, 28, 1};
  iree_hal_element_type_t hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  float input_range[2] = {0.0f, 1.0f};
  IREE_RETURN_IF_ERROR(
      iree_tools_utils_buffer_view_from_image_rescaled(
          image_path, buffer_shape, IREE_ARRAYSIZE(buffer_shape),
          hal_element_type, iree_hal_device_allocator(device), input_range,
          IREE_ARRAYSIZE(input_range), &buffer_view),
      "load image");
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view));
  iree_hal_buffer_view_release(buffer_view);

  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret_buffer_view));

  // Read back the results. The output of the mnist model is a 1x10 prediction
  // confidence values for each digit in [0, 9].
  float predictions[1 * 10] = {0.0f};
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session),
      iree_hal_buffer_view_buffer(ret_buffer_view), 0, predictions,
      sizeof(predictions), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  iree_hal_buffer_view_release(ret_buffer_view);

  // Get the highest index from the output.
  float result_val = FLT_MIN;
  int result_idx = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(predictions); ++i) {
    if (predictions[i] > result_val) {
      result_val = predictions[i];
      result_idx = i;
    }
  }
  fprintf(stdout, "Detected number: %d\n", result_idx);

  iree_runtime_call_deinitialize(&call);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  if (argc > 2) {
    fprintf(stderr, "Usage: iree-run-mnist-module <image file>\n");
    return -1;
  }
  iree_string_view_t image_path;
  if (argc == 1) {
    image_path = iree_make_cstring_view("mnist_test.png");
  } else {
    image_path = iree_make_cstring_view(argv[1]);
  }
  iree_status_t result = Run(image_path);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  iree_status_ignore(result);
  return 0;
}
