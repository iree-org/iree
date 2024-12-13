// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir::iree_compiler;
using namespace testing;

TEST(Permutation, MakeMovePermutation) {
  EXPECT_THAT(makeMovePermutation(1, 0, 0), ElementsAre(0));
  EXPECT_THAT(makeMovePermutation(2, 0, 1), ElementsAre(1, 0));
  EXPECT_THAT(makeMovePermutation(5, 1, 3), ElementsAre(0, 2, 3, 1, 4));
  EXPECT_THAT(makeMovePermutation(3, 1, 2), ElementsAre(0, 2, 1));
  EXPECT_THAT(makeMovePermutation(3, 2, 0), ElementsAre(2, 0, 1));
}

TEST(EmbeddedDataDirectory, AddFileGetFile) {
  EmbeddedDataDirectory dir;
  EXPECT_TRUE(dir.addFile("filename1", "file contents 1"));
  EXPECT_TRUE(dir.addFile("filename2", "file contents 2"));
  EXPECT_FALSE(dir.addFile("filename1", "file contents 3"));
  EXPECT_EQ(dir.getFile("filename1"), "file contents 1");
  EXPECT_EQ(dir.getFile("filename2"), "file contents 2");
  EXPECT_EQ(dir.getFile("filename3"), std::nullopt);
}

TEST(EmbeddedDataDirectory, WithGlobal) {
  std::vector<std::thread> threads;
  for (int i = 0; i < 3; ++i) {
    threads.emplace_back([i] {
      EmbeddedDataDirectory::withGlobal([i](EmbeddedDataDirectory &globalDir) {
        EXPECT_TRUE(globalDir.addFile(llvm::formatv("filename{}", i).str(),
                                      "file contents xxx"));
      });
    });
  }
  for (std::thread &thread : threads) {
    thread.join();
  }
  EmbeddedDataDirectory::withGlobal([](EmbeddedDataDirectory &globalDir) {
    std::vector<std::string> keys;
    for (auto iter : globalDir.getMap().keys()) {
      keys.push_back(iter.str());
    }
    EXPECT_THAT(keys,
                UnorderedElementsAre("filename0", "filename1", "filename2"));
  });
}

TEST(EmbeddedDataDirectory, GetMap) {
  EmbeddedDataDirectory dir;
  EXPECT_TRUE(dir.addFile("filename1", "file contents 1"));
  EXPECT_TRUE(dir.addFile("filename2", "file contents 2"));
  std::vector<std::string> keys;
  for (auto iter : dir.getMap().keys()) {
    keys.push_back(iter.str());
  }
  EXPECT_THAT(keys, UnorderedElementsAre("filename1", "filename2"));
}
