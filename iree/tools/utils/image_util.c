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
#include "iree/tools/utils/image_util.h"

#include <math.h>

#include "iree/base/internal/flags.h"
#include "stb_image.h"

IREE_FLAG(string, input_range, "0 1",
          "Float input dynamic range separated by the space or comma. "
          "Default to \"0 1\"");

// Parse the FLAG_input_range to the unit8 --> float scaling parameters.
// float32_x = (uint8_x - 127.5) / 127.5 * input_scale[0] + input_scale[1]
iree_status_t iree_tools_utils_parse_pixel_range(float* input_scale) {
  iree_string_view_t input_range_str = iree_make_cstring_view(FLAG_input_range);
  int idx = iree_string_view_find_first_of(input_range_str,
                                           iree_make_cstring_view(" ,"), 0);
  if (idx <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "parse input range");
  }
  float input_range[2];
  if (!iree_string_view_atof(iree_string_view_substr(input_range_str, 0, idx),
                             &input_range[0]) ||
      !iree_string_view_atof(
          iree_string_view_substr(input_range_str, idx + 1, INTPTR_MAX),
          &input_range[1])) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "parse input range");
  }
  input_scale[0] = fabsf(input_range[1] - input_range[0]) / 2.0f;
  input_scale[1] = (input_range[0] + input_range[1]) / 2.0f;
  return iree_ok_status();
}

iree_status_t iree_tools_utils_pixel_rescaled_to_buffer(
    const uint8_t* pixel_data, iree_host_size_t buffer_length, float* buffer) {
  float input_scale[2] = {0.0f};
  IREE_RETURN_IF_ERROR(iree_tools_utils_parse_pixel_range(input_scale));
  const float kUint8Mean = 127.5f;
  for (int i = 0; i < buffer_length; ++i) {
    buffer[i] =
        (((float)(pixel_data[i])) - kUint8Mean) / kUint8Mean * input_scale[0] +
        input_scale[1];
  }
  return iree_ok_status();
}

iree_status_t iree_tools_utils_load_pixel_data(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, const char* filename,
    uint8_t** pixel_data, iree_host_size_t* buffer_length) {
  int img_dims[3];
  if (stbi_info(filename, img_dims, &(img_dims[1]), &(img_dims[2])) == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "can't load image %s",
                            filename);
  }
  if (!(element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
        element_type == IREE_HAL_ELEMENT_TYPE_SINT_8 ||
        element_type == IREE_HAL_ELEMENT_TYPE_UINT_8)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "element type not supported");
  }
  switch (shape_rank) {
    case 2: {  // Assume tensor <heightxwidth>
      if (img_dims[2] != 1 || (shape[0] != img_dims[1]) ||
          (shape[1] != img_dims[0])) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "image size: %dx%dx%d, expected: %dx%d",
                                img_dims[0], img_dims[1], img_dims[2], shape[1],
                                shape[0]);
      }
      break;
    }
    case 3: {  // Assume tensor <heightxwidthxchannel>
      if (shape[0] != img_dims[1] || shape[1] != img_dims[0] ||
          shape[2] != img_dims[2]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "image size: %dx%dx%d, expected: %dx%dx%d",
                                img_dims[0], img_dims[1], img_dims[2], shape[1],
                                shape[0], shape[2]);
      }
      break;
    }
    case 4: {  // Assume tensor <batchxheightxwidthxchannel>
      if (shape[1] != img_dims[1] || shape[2] != img_dims[0] ||
          shape[3] != img_dims[2]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "image size: %dx%dx%d, expected: %dx%dx%d",
                                img_dims[0], img_dims[1], img_dims[2], shape[2],
                                shape[1], shape[3]);
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Input buffer shape rank %lu not supported",
                              shape_rank);
  }
  // Drop the alpha channel.
  int req_ch = (img_dims[2] >= 3) ? 3 : 0;
  *pixel_data =
      stbi_load(filename, img_dims, &(img_dims[1]), &(img_dims[2]), req_ch);
  if (*pixel_data == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "can't load image %s",
                            filename);
  }
  *buffer_length =
      img_dims[0] * img_dims[1] * (img_dims[2] > 3 ? 3 : img_dims[2]);
  return iree_ok_status();
}

iree_status_t iree_tools_utils_buffer_view_from_image(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, iree_hal_allocator_t* allocator,
    const char* filename, iree_hal_buffer_view_t** buffer_view) {
  *buffer_view = NULL;
  iree_status_t result;
  uint8_t* pixel_data = NULL;
  iree_host_size_t buffer_length;
  result = iree_tools_utils_load_pixel_data(
      shape, shape_rank, element_type, filename, &pixel_data, &buffer_length);
  if (!iree_status_is_ok(result)) {
    return result;
  }

  iree_hal_buffer_t* buffer = NULL;
  iree_host_size_t element_byte = iree_hal_element_byte_count(element_type);
  switch (element_type) {
    // SINT_8 and UINT_8 perform direct buffer wrap.
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8: {
      result = iree_hal_buffer_view_wrap_or_clone_heap_buffer(
          allocator, shape, shape_rank, element_type,
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_USAGE_ALL,
          iree_make_byte_span((void*)pixel_data, element_byte * buffer_length),
          iree_allocator_null(), buffer_view);
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: {
      result = iree_hal_allocator_allocate_buffer(
          allocator,
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          IREE_HAL_BUFFER_USAGE_ALL, element_byte * buffer_length, &buffer);
      if (!iree_status_is_ok(result)) {
        free(pixel_data);
        return result;
      }
      // Need to normalize to the expected input range. Default to [0, 1].
      iree_hal_buffer_mapping_t mapped_memory;
      result = iree_hal_buffer_map_range(
          buffer, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0,
          element_byte * buffer_length, &mapped_memory);
      if (!iree_status_is_ok(result)) {
        iree_hal_buffer_release(buffer);
        free(pixel_data);
        return result;
      }
      result = iree_tools_utils_pixel_rescaled_to_buffer(
          pixel_data, buffer_length, (float*)mapped_memory.contents.data);
      if (!iree_status_is_ok(result)) {
        iree_hal_buffer_release(buffer);
        free(pixel_data);
        return result;
      }
      iree_hal_buffer_unmap_range(&mapped_memory);
      result = iree_hal_buffer_view_create(buffer, shape, shape_rank,
                                           element_type, buffer_view);
      iree_hal_buffer_release(buffer);
      break;
    }
    default: {
      free(pixel_data);
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "element type not supported");
    }
  }
  free(pixel_data);
  return result;
}
