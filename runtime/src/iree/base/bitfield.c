// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/bitfield.h"

#include <stdlib.h>
#include <string.h>

IREE_API_EXPORT iree_status_t
iree_bitfield_format(uint32_t value, iree_host_size_t mapping_count,
                     const iree_bitfield_string_mapping_t* mappings,
                     iree_string_builder_t* string_builder) {
  uint32_t remaining_bits = value;
  int i = 0;
  for (iree_host_size_t mapping_index = 0; mapping_index < mapping_count;
       ++mapping_index) {
    const iree_bitfield_string_mapping_t mapping = mappings[mapping_index];
    if ((remaining_bits & mapping.bits) == mapping.bits) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(string_builder, IREE_SV("|")));
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(string_builder, mapping.string));
      remaining_bits &= ~mapping.bits;
      ++i;
    }
  }
  if (remaining_bits != 0u) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(string_builder, IREE_SV("|")));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        string_builder, "%Xh", remaining_bits));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_string_view_t
iree_bitfield_format_inline(uint32_t value, iree_host_size_t mapping_count,
                            const iree_bitfield_string_mapping_t* mappings,
                            iree_bitfield_string_temp_t* out_temp) {
  iree_string_builder_t string_builder;
  iree_string_builder_initialize_with_storage(
      out_temp->buffer, IREE_ARRAYSIZE(out_temp->buffer), &string_builder);
  iree_status_t status =
      iree_bitfield_format(value, mapping_count, mappings, &string_builder);
  if (iree_status_is_ok(status)) {
    return iree_string_builder_view(&string_builder);
  }
  iree_status_ignore(status);
  return IREE_SV("(error)");
}
