// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_COMMON_COMMAND_LINE_UTILS_H_
#define IREE_PJRT_PLUGIN_PJRT_COMMON_COMMAND_LINE_UTILS_H_

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace iree {
namespace pjrt {

// parse command line options (maybe with quotes) to an array of options
// e.g. `a b "c d"` -> {"a", "b", "c d"}
std::optional<std::vector<std::string>> ParseOptionsFromCommandLine(
    std::string_view options_str);

}  // namespace pjrt
}  // namespace iree

#endif
