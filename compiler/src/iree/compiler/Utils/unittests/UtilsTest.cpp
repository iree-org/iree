// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "iree/compiler/Utils/Indexing.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
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

TEST(BasisFromSizeStrides, SimpleCase) {
  SmallVector<int64_t> basis;
  SmallVector<size_t> dimToResult;

  EXPECT_TRUE(
      succeeded(basisFromSizesStrides({4, 16}, {1, 4}, basis, dimToResult)));
  EXPECT_THAT(basis, ElementsAre(16, 4));
  EXPECT_THAT(dimToResult, ElementsAre(2, 1));
}

TEST(BasisFromSizeStrides, ZeroStride) {
  SmallVector<int64_t> basis;
  SmallVector<size_t> dimToResult;

  EXPECT_TRUE(succeeded(
      basisFromSizesStrides({16, 4, 4}, {1, 0, 16}, basis, dimToResult)));
  EXPECT_THAT(basis, ElementsAre(4, 16, 1));
  EXPECT_THAT(dimToResult, ElementsAre(2, 3, 1));
}

TEST(BasisFromSizeStrides, JumpsInStrides) {
  SmallVector<int64_t> basis;
  SmallVector<size_t> dimToResult;

  EXPECT_TRUE(
      succeeded(basisFromSizesStrides({8, 4}, {8, 1}, basis, dimToResult)));
  EXPECT_THAT(basis, ElementsAre(8, 2, 4));
  EXPECT_THAT(dimToResult, ElementsAre(1, 3));
}

TEST(BasisFromSizeStrides, OverlappingStrides) {
  SmallVector<int64_t> basis;
  SmallVector<size_t> dimToResult;

  EXPECT_FALSE(
      succeeded(basisFromSizesStrides({8, 4}, {6, 1}, basis, dimToResult)));
}

//=------------------------------------------------------------------------------=//
// OptionUtils tests
//=------------------------------------------------------------------------------=//

namespace {
struct TestOptions {
  llvm::OptimizationLevel rootOption = llvm::OptimizationLevel::O0;
  llvm::OptimizationLevel parentOption = llvm::OptimizationLevel::O0;
  bool childOption = false;
  static constexpr llvm::StringRef kRootOptFlag = "root-option";

  void bindOptions(OptionsBinder &binder) {
    binder.topLevelOpt(kRootOptFlag, rootOption);
    auto init_at_opt = binder.optimizationLevel("parent-option", parentOption);
    binder.opt<bool>("child-option", childOption,
                     {init_at_opt(llvm::OptimizationLevel::O3, true)},
                     llvm::cl::desc("test desc."));
  }
};
} // namespace

TEST(OptionUtils, DefaultTest) {
  auto binder = OptionsBinder::local();
  TestOptions opts;
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.childOption, false);

  opts.bindOptions(binder);
  LogicalResult parseResult = binder.parseArguments(0, nullptr);

  EXPECT_TRUE(succeeded(parseResult));
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.childOption, false);
}

TEST(OptionUtils, OverrideParent) {
  auto binder = OptionsBinder::local();
  TestOptions opts;
  opts.bindOptions(binder);

  int argc = 1;
  const char *argv[] = {"--parent-option=O3"};
  LogicalResult parseResult = binder.parseArguments(argc, argv);
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.childOption, false);

  binder.applyOptimizationDefaults();
  EXPECT_TRUE(succeeded(parseResult));
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.childOption, true);

  binder.restoreOptimizationDefaults();
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.childOption, false);
}

TEST(OptionUtils, OverrideRoot) {
  auto binder = OptionsBinder::local();
  TestOptions opts;
  opts.bindOptions(binder);

  int argc = 1;
  const char *argv[] = {"--root-option=O3"};
  LogicalResult parseResult = binder.parseArguments(argc, argv);

  binder.applyOptimizationDefaults();
  EXPECT_TRUE(succeeded(parseResult));
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.childOption, true);

  binder.restoreOptimizationDefaults();
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.childOption, false);
}

TEST(OptionUtils, OverrideRootAndSetChild) {
  auto binder = OptionsBinder::local();
  TestOptions opts;
  opts.bindOptions(binder);

  int argc = 2;
  const char *argv[] = {"--root-option=O3", "--child-option=true"};
  LogicalResult parseResult = binder.parseArguments(argc, argv);
  EXPECT_TRUE(succeeded(parseResult));
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.childOption, true);

  binder.applyOptimizationDefaults();
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.childOption, true);

  binder.restoreOptimizationDefaults();
  EXPECT_EQ(opts.rootOption, llvm::OptimizationLevel::O3);
  EXPECT_EQ(opts.parentOption, llvm::OptimizationLevel::O0);
  EXPECT_EQ(opts.childOption, true);
}
