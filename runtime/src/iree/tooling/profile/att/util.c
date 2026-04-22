// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/util.h"

#include <string.h>

iree_status_t iree_profile_att_grow_array(iree_allocator_t host_allocator,
                                          iree_host_size_t element_count,
                                          iree_host_size_t element_size,
                                          iree_host_size_t* inout_capacity,
                                          void** inout_ptr) {
  if (element_count <= *inout_capacity) return iree_ok_status();
  return iree_allocator_grow_array(
      host_allocator, iree_max((iree_host_size_t)16, element_count),
      element_size, inout_capacity, inout_ptr);
}

iree_status_t iree_profile_att_copy_cstring(iree_string_view_t value,
                                            iree_allocator_t host_allocator,
                                            char** out_string) {
  *out_string = NULL;
  char* string = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, value.size + 1, (void**)&string));
  if (value.size > 0) memcpy(string, value.data, value.size);
  string[value.size] = 0;
  *out_string = string;
  return iree_ok_status();
}

iree_string_view_t iree_profile_att_cstring_view_or_empty(const char* string) {
  return string ? iree_make_cstring_view(string) : iree_string_view_empty();
}
