// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/testing/test_runner.h"

namespace iree::vm::testing {

std::ostream& operator<<(std::ostream& os, const VMTestParams& params) {
  std::string name = params.module_name + "_" + params.function_name;
  // Replace special characters for valid test names.
  for (char& c : name) {
    if (c == ':' || c == '.') c = '_';
  }
  return os << name;
}

}  // namespace iree::vm::testing
