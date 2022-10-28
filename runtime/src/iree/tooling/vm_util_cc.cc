// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/vm_util_cc.h"

#include <vector>

#include "iree/vm/api.h"

namespace iree {

Status ParseToVariantList(iree_hal_allocator_t* device_allocator,
                          iree::span<const std::string> input_strings,
                          iree_allocator_t host_allocator,
                          iree_vm_list_t** out_list) {
  std::vector<iree_string_view_t> input_string_views(input_strings.size());
  for (size_t i = 0; i < input_strings.size(); ++i) {
    input_string_views[i].data = input_strings[i].data();
    input_string_views[i].size = input_strings[i].size();
  }
  return iree_tooling_parse_to_variant_list(
      device_allocator, input_string_views.data(), input_string_views.size(),
      host_allocator, out_list);
}

Status PrintVariantList(iree_vm_list_t* variant_list, size_t max_element_count,
                        std::string* out_string) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_RETURN_IF_ERROR(iree_tooling_append_variant_list_lines(
      variant_list, max_element_count, &builder));
  out_string->assign(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return iree_ok_status();
}

}  // namespace iree
