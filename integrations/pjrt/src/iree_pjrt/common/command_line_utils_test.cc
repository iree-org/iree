// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/common/command_line_utils.h"

#include <gtest/gtest.h>

using namespace iree::pjrt;

TEST(CommandLineUtils, ParseOptionsFromCommandLine) {
  EXPECT_EQ(ParseOptionsFromCommandLine("--help --verbose"),
            (std::vector<std::string>{"--help", "--verbose"}));
  EXPECT_EQ(ParseOptionsFromCommandLine("-a='x y' -b \"n m\""),
            (std::vector<std::string>{"-a=x y", "-b", "n m"}));
  EXPECT_EQ(ParseOptionsFromCommandLine("'\"' \"'\""),
            (std::vector<std::string>{"\"", "'"}));
  EXPECT_EQ(ParseOptionsFromCommandLine("ab   abc d 'e f g' h  "),
            (std::vector<std::string>{"ab", "abc", "d", "e f g", "h"}));
  EXPECT_EQ(ParseOptionsFromCommandLine("a 'b"), std::nullopt);
  EXPECT_EQ(ParseOptionsFromCommandLine("x\"y"), std::nullopt);
}
