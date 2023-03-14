// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/embed_data/testembed1.h"
#include "build_tools/embed_data/testembed2.h"
#include "iree/testing/gtest.h"

namespace {

#if defined(_MSC_VER)
#define NEWLINE "\r\n"
#else
#define NEWLINE "\n"
#endif

TEST(Generator, TestContents) {
  auto* toc1 = testembed1_create();
  ASSERT_EQ("file1.bin", std::string(toc1->name));
  EXPECT_EQ(R"(Are you '"Still"' here?)" NEWLINE, std::string(toc1->data));
#if defined(_MSC_VER)
  EXPECT_EQ(25, toc1->size);
#else
  EXPECT_EQ(24, toc1->size);
#endif
  ASSERT_EQ(0, *(toc1->data + toc1->size));

  ++toc1;
  ASSERT_EQ("file2.bin", std::string(toc1->name));
  EXPECT_EQ(R"(¯\_(ツ)_/¯)" NEWLINE, std::string(toc1->data));
#if defined(_MSC_VER)
  EXPECT_EQ(15, toc1->size);
#else
  ASSERT_EQ(14, toc1->size);
#endif
  ASSERT_EQ(0, *(toc1->data + toc1->size));

  ++toc1;
  EXPECT_EQ(nullptr, toc1->name);
  ASSERT_EQ(nullptr, toc1->data);

  auto* toc2 = testembed2_create();
  ASSERT_EQ("file3.bin", std::string(toc2->name));
  EXPECT_EQ(R"(ᕕ( ᐛ )ᕗ)" NEWLINE, std::string(toc2->data));
#if defined(_MSC_VER)
  EXPECT_EQ(15, toc2->size);
#else
  EXPECT_EQ(14, toc2->size);
#endif
  ASSERT_EQ(0, *(toc2->data + toc2->size));
}

}  // namespace
