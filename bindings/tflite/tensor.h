// Copyright 2020 Google LLC
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

#ifndef IREE_BINDINGS_TFLITE_TENSOR_H_
#define IREE_BINDINGS_TFLITE_TENSOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1  // force dllexport
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

// This is the same value as TFLITE_RESHAPE_PARAMS_MAX_DIMENSION_COUNT.
// Since that's all tflite supports internally we are fairly safe to use that
// as our max for the I/O tensors; internal tensors inside of IREE generated
// modules may of course have arbitrary shape ranks.
#define IREE_BINDINGS_TFLITE_MAX_RANK 8

struct TfLiteTensor {
  // Static metadata about the tensor as it was embedded in the module.
  TfLiteType type;
  TfLiteQuantizationParams quantization_params;
  iree_string_view_t name;

  // Queried shape information from the module; in the case of outputs this is
  // the expected output shape based on the current input shapes and will be
  // available even if we haven't yet run and allocated the buffer view.
  int32_t shape_rank;
  int32_t shape_dims[IREE_BINDINGS_TFLITE_MAX_RANK];

  // Allocated buffer view referencing the backing tensor memory.
  iree_hal_buffer_t* buffer;
  // Persistently mapped buffer; invalidated when buffer is resized.
  iree_hal_buffer_mapping_t buffer_mapping;
};

// Parses a tfl.io.names value and sets the |tensor| name.
iree_status_t _TfLiteTensorParseNameAttr(TfLiteTensor* tensor,
                                         iree_string_view_t attr,
                                         iree_allocator_t allocator);

// Parses a tfl.io.types value and sets the |tensor| type.
iree_status_t _TfLiteTensorParseTypeAttr(TfLiteTensor* tensor,
                                         iree_string_view_t attr);

// Parses a tfl.io.quant value and sets the |tensor| quantization parameters.
iree_status_t _TfLiteTensorParseQuantAttr(TfLiteTensor* tensor,
                                          iree_string_view_t attr);

// Reallocates and remaps the tensor buffer view if needed.
// No-op if the buffer view is already allocated and its shape matches the
// current tensor shape.
iree_status_t _TfLiteTensorReallocateIfNeeded(
    TfLiteTensor* tensor, iree_hal_allocator_t* buffer_allocator,
    iree_allocator_t heap_allocator);

// Binds the given |buffer| to the tensor and maps it.
// The tensor shape will be overwritten with the buffer view shape.
iree_status_t _TfLiteTensorBind(TfLiteTensor* tensor,
                                iree_hal_buffer_t* buffer);

// Discards the current buffer view, if any, resetting it to NULL.
void _TfLiteTensorDiscardBuffer(TfLiteTensor* tensor);

// Resets the tensor back to its initial state (no buffers, etc).
void _TfLiteTensorReset(TfLiteTensor* tensor, iree_allocator_t allocator);

#endif  // IREE_BINDINGS_TFLITE_TENSOR_H_
