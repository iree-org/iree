// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command_line_utils.h"

namespace iree {
namespace pjrt {

// TODO: currently this function doesn't handle escape sequences,
// it just ensure that single/double quotes are interpreted corrently.
std::optional<std::vector<std::string>> ParseOptionsFromCommandLine(
    std::string_view options_str) {
  std::vector<std::string> options;
  std::string current;

  enum { NORMAL, SINGLE_QUOTE, DOUBLE_QUOTE } state = NORMAL;
  for (auto it = options_str.begin(); it != options_str.end(); ++it) {
    if (std::isspace(*it) && state == NORMAL) {
      if (!current.empty()) {
        options.push_back(std::move(current));
        current.clear();
      }
    } else if (*it == '"' && state != SINGLE_QUOTE) {
      if (state == NORMAL)
        state = DOUBLE_QUOTE;
      else if (state == DOUBLE_QUOTE)
        state = NORMAL;
    } else if (*it == '\'' && state != DOUBLE_QUOTE) {
      if (state == NORMAL)
        state = SINGLE_QUOTE;
      else if (state == SINGLE_QUOTE)
        state = NORMAL;
    } else {
      current.push_back(*it);
    }
  }

  if (!current.empty()) {
    options.push_back(std::move(current));
  }

  // if it's still in a quote, then return nullopt
  if (state != NORMAL) {
    return std::nullopt;
  }

  return options;
}

}  // namespace pjrt
}  // namespace iree
