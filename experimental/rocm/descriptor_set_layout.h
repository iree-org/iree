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

#ifndef IREE_HAL_ROCM_DESCRIPTOR_SET_LAYOUT_H_
#define IREE_HAL_ROCM_DESCRIPTOR_SET_LAYOUT_H_

#include "experimental/rocm/context_wrapper.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_rocm_descriptor_set_layout_create(
    iree_hal_rocm_context_wrapper_t *context,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t *bindings,
    iree_hal_descriptor_set_layout_t **out_descriptor_set_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_DESCRIPTOR_SET_LAYOUT_H_
