// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/hal/buffer_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace iree::pjrt {

class ApiMemoryLayout {
 public:
  ApiMemoryLayout() = default;

  void InitializeDenseRowMajorStrided(size_t rank, const int64_t *dims,
                                      size_t unit_stride_bytes);
  void InitializeDenseRowMajorTiled(int64_t rank);
  void Reset() { valid_ = false; }

  bool is_valid() const { return valid_; }
  const PJRT_Buffer_MemoryLayout &c_layout() const { return c_layout_; };

 private:
  PJRT_Buffer_MemoryLayout c_layout_;

  // Retained vector of ints.
  std::vector<int64_t> storage1_;

  bool valid_ = false;
};

}  // namespace iree::pjrt
