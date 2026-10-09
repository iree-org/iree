// Copyright 2026 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include <string>
#include "gtest/gtest.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {
namespace {
constexpr StringLiteral kLoop = R"mlir(
func.func @loop(%input: memref<4xi32>, %count: index) -> memref<4xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tensor = bufferization.to_tensor %input restrict writable : memref<4xi32> to tensor<4xi32>
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %tensor) -> tensor<4xi32> {
    %copy = bufferization.alloc_tensor() copy(%iter) : tensor<4xi32>
    scf.yield %copy : tensor<4xi32>
  }
  %buffer = bufferization.to_buffer %result : tensor<4xi32> to memref<4xi32>
  return %buffer : memref<4xi32>
}
)mlir";

class BufferizePassesTest : public ::testing::Test {
protected:
  BufferizePassesTest() {
    DialectRegistry registry;
    registry
        .insert<arith::ArithDialect, bufferization::BufferizationDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
    registerBufferizationInterfaces(registry);
    context.appendDialectRegistry(registry);
  }
  MLIRContext context;
};

TEST_F(BufferizePassesTest, RejectsLoopAllocationsByDefault) {
  auto module = parseSourceString<ModuleOp>(kLoop, &context);
  ASSERT_TRUE(module);
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    diagnostic.print(stream);
    return success();
  });
  PassManager manager(&context);
  addIREEComprehensiveBufferizePasses(manager.nest<func::FuncOp>());
  EXPECT_TRUE(failed(manager.run(*module)));
  EXPECT_NE(diagnostics.find("not equivalent to the corresponding iter bbArg"),
            std::string::npos)
      << diagnostics;
}

TEST_F(BufferizePassesTest, AllowsLoopAllocationsWithCustomCopy) {
  auto module = parseSourceString<ModuleOp>(kLoop, &context);
  ASSERT_TRUE(module);
  auto copy = [](OpBuilder &builder, Location loc, Value from, Value to) {
    memref::CopyOp::create(builder, loc, from, to);
    return success();
  };
  PassManager manager(&context);
  addIREEComprehensiveBufferizePasses(manager.nest<func::FuncOp>(),
                                      /*allocationFn=*/std::nullopt, copy,
                                      /*allowReturnAllocsFromLoops=*/true);
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  scf::ForOp loop;
  module->walk([&](scf::ForOp op) { loop = op; });
  ASSERT_TRUE(loop);
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto allocation = yield.getOperand(0).getDefiningOp<memref::AllocOp>();
  ASSERT_TRUE(allocation) << "Loop must yield its new allocation";
  EXPECT_EQ(allocation->getBlock(), loop.getBody());
  auto copies = loop.getBody()->getOps<memref::CopyOp>();
  ASSERT_EQ(std::distance(copies.begin(), copies.end()), 1);
  auto copyOp = *copies.begin();
  EXPECT_EQ(copyOp.getSource(), loop.getRegionIterArgs().front());
  EXPECT_EQ(copyOp.getTarget(), allocation.getResult());
}
} // namespace
} // namespace mlir::iree_compiler
