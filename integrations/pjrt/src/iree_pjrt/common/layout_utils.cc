// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/common/layout_utils.h"

#include <cstring>

namespace iree::pjrt {

void ApiMemoryLayout::InitializeDenseRowMajorStrided(size_t rank,
                                                     const int64_t *dims,
                                                     size_t unit_stride_bytes) {
  memset(&c_layout_, 0, sizeof(c_layout_));
  int64_t stride = unit_stride_bytes;
  storage1_.resize(rank);
  for (size_t pos = 0; pos < rank; ++pos) {
    storage1_[rank - pos - 1] = stride;
    stride *= dims[pos];
  }

  c_layout_.struct_size = sizeof(c_layout_);
  c_layout_.type = PJRT_Buffer_MemoryLayout_Type_Strides;
  c_layout_.strides.struct_size = sizeof(c_layout_.strides);
  c_layout_.strides.byte_strides = storage1_.data();
  c_layout_.strides.num_byte_strides = storage1_.size();
  valid_ = true;
}

void ApiMemoryLayout::InitializeDenseRowMajorTiled(int64_t rank) {
  memset(&c_layout_, 0, sizeof(c_layout_));
  // Set minor_to_major. See SetDefaultLayoutToContainer in LayoutUtil.h
  storage1_.resize(rank, 0);
  for (int64_t i = 0; i < rank; ++i) {
    storage1_[i] = rank - 1 - i;
  }
  c_layout_.struct_size = sizeof(c_layout_);
  c_layout_.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
  c_layout_.tiled.struct_size = sizeof(c_layout_.tiled);
  c_layout_.tiled.minor_to_major = storage1_.data();
  c_layout_.tiled.minor_to_major_size = storage1_.size();
  valid_ = true;
}

}  // namespace iree::pjrt
