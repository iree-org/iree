// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_TOOLS_UTILS_IMAGE_UTIL_H_
#define IREE_TOOLS_UTILS_IMAGE_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/buffer_view.h"

#if __cplusplus
extern "C" {
#endif  // __cplusplus

// Check the hal buffer information with the image width, height, and channel
// number. Try to load the image in |filename| with stb_image.h support.
// Function returns a populated pixel_data buffer and the buffer length.
// The returned pixel_data buffer must be released by the caller.
iree_status_t iree_tools_utils_load_pixel_data(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, const char* filename,
    uint8_t** pixel_data, iree_host_size_t* buffer_length);

// Parse the content in an image file in |filename| into a HAL buffer view
// using stb library (stb_image.h).
// The returned buffer_view pointer must be released by the caller.
iree_status_t iree_tools_utils_buffer_view_from_image(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, iree_hal_allocator_t* allocator,
    const char* filename, iree_hal_buffer_view_t** buffer_view);

// Normalize pixel data to a buffer of floating number based on the input range.
// The output buffer needs to be allocated before the call.
iree_status_t iree_tools_utils_pixel_rescaled_to_buffer(
    const uint8_t* pixel_data, iree_host_size_t buffer_length, float* buffer);

#if __cplusplus
}
#endif  // __cplusplus

#endif  // IREE_TOOLS_UTILS_IMAGE_UTIL_H_
