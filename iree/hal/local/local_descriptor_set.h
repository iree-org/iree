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

#ifndef IREE_HAL_LOCAL_LOCAL_DESCRIPTOR_SET_H_
#define IREE_HAL_LOCAL_LOCAL_DESCRIPTOR_SET_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/local_descriptor_set_layout.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_local_descriptor_set_layout_t* layout;
  iree_host_size_t binding_count;
  iree_hal_descriptor_set_binding_t bindings[];
} iree_hal_local_descriptor_set_t;

iree_status_t iree_hal_local_descriptor_set_create(
    iree_hal_descriptor_set_layout_t* layout, iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set);

iree_hal_local_descriptor_set_t* iree_hal_local_descriptor_set_cast(
    iree_hal_descriptor_set_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_DESCRIPTOR_SET_H_
