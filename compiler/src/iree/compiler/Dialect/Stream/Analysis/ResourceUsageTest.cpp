// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/ResourceUsage.h"

#include "gtest/gtest.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::IREE::Stream {
namespace {

class MutatingTiedUsesTest : public ::testing::Test {
protected:
  MutatingTiedUsesTest() {
    context.loadDialect<StreamDialect, IREE::Util::UtilDialect,
                        arith::ArithDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, &context);
  }

  // Query directly: the surrounding DFX solver's 32-iteration limit is a
  // separate contract and must not mask the traversal's stack-depth test.
  void checkDeepChain(bool mutate) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "util.func private @chain(%source: !stream.resource<constant>, "
          "%size: index) -> !stream.resource<*> {\n"
          "  %zero = arith.constant 0 : index\n"
          "  %one = arith.constant 1 : i32\n"
          "  %clone = stream.async.clone %source : "
          "!stream.resource<constant>{%size} -> !stream.resource<*>{%size}\n";
    std::string previous = "%clone";
    for (int i = 0; i < 10000; ++i) {
      os << "  %b" << i << " = util.optimization_barrier " << previous
         << " : !stream.resource<*>\n";
      previous = "%b" + std::to_string(i);
    }
    if (mutate) {
      os << "  %filled = stream.async.fill %one, " << previous
         << "[%zero to %size for %size] : i32 -> " << previous
         << " as !stream.resource<*>{%size}\n";
      previous = "%filled";
    }
    os << "  util.return " << previous << " : !stream.resource<*>\n}\n";
    auto module = parse(source);
    ASSERT_TRUE(module);
    module->walk([&](AsyncCloneOp clone) {
      EXPECT_EQ(hasMutatingTiedUses(clone.getResult()), mutate);
    });
  }

  MLIRContext context;
};

TEST_F(MutatingTiedUsesTest, DeepMutatingChain) { checkDeepChain(true); }
TEST_F(MutatingTiedUsesTest, DeepReadonlyChain) { checkDeepChain(false); }

TEST_F(MutatingTiedUsesTest, VariadicSlotsAndUnknownTiedUse) {
  auto module = parse(R"mlir(
    util.func private @unknown(%arg: !stream.resource<*>) -> %arg
    util.func private @slots(%source: !stream.resource<constant>, %size: index) {
      %left = stream.async.clone %source : !stream.resource<constant>{%size} -> !stream.resource<*>{%size}
      %right = stream.async.clone %source : !stream.resource<constant>{%size} -> !stream.resource<*>{%size}
      %values:2 = util.optimization_barrier %left, %right : !stream.resource<*>, !stream.resource<*>
      %result = util.call @unknown(%values#1) : (!stream.resource<*>) -> %values#1
      util.return
    }
  )mlir");
  ASSERT_TRUE(module);
  SmallVector<AsyncCloneOp> clones;
  module->walk([&](AsyncCloneOp clone) { clones.push_back(clone); });
  ASSERT_EQ(clones.size(), 2u);
  EXPECT_FALSE(hasMutatingTiedUses(clones[0].getResult()));
  EXPECT_TRUE(hasMutatingTiedUses(clones[1].getResult()));
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Stream
