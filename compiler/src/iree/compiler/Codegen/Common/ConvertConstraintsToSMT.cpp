// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------- ConvertConstraintsToSMT.cpp ---------------------===//
//
// Converts IREE Codegen constraint op to mlir::smt::SolverOp.
//
// The pass rewrites the constraints body in-place with the SMT ops, block
// arguments are preserved to pass the ConstraintsOp verifier.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTCONSTRAINTSTOSMTPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static smt::SolverOp
convertConstraintsToSMTSolver(IREE::Codegen::ConstraintsOp op,
                              OpBuilder &builder) {
  Location loc = op.getLoc();

  auto solverOp =
      smt::SolverOp::create(builder, loc, TypeRange{}, ValueRange{});

  Block *solverBody = &solverOp.getBodyRegion().emplaceBlock();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(solverBody);

  // Map ConstraintsOp's block arguments to smt.declare_fun ops.
  IRMapping mapping;
  Block &constraintsBody = op.getBody().front();
  for (auto [idx, blockArg] : llvm::enumerate(constraintsBody.getArguments())) {
    auto declareFun = smt::DeclareFunOp::create(
        builder, loc, blockArg.getType(),
        builder.getStringAttr("problem_dim_" + llvm::Twine(idx)));
    mapping.map(blockArg, declareFun.getResult());
  }

  for (Operation &bodyOp : constraintsBody) {
    TypeSwitch<Operation *>(&bodyOp)
        .Case([&](IREE::Codegen::AssertOp assertOp) {
          Value mappedCond = mapping.lookupOrDefault(assertOp.getCondition());
          smt::AssertOp::create(builder, loc, mappedCond);
        })
        .Case([&](IREE::Codegen::KnobOp knobOp) {
          auto declareFun = smt::DeclareFunOp::create(
              builder, loc, knobOp.getResult().getType(), knobOp.getNameAttr());
          mapping.map(knobOp.getResult(), declareFun.getResult());
        })
        .Case([&](IREE::Codegen::LookupOp lookupOp) {
          Value mappedIdx = mapping.lookupOrDefault(lookupOp.getIndex());
          auto keys = lookupOp.getKeys();
          auto vals = lookupOp.getValues();

          // vals is non-empty (guaranteed by LookupOp::verify).
          Value result = smt::IntConstantOp::create(
              builder, loc, builder.getI64IntegerAttr(vals.back()));

          for (auto [key, val] : llvm::reverse(
                   llvm::zip_equal(keys.drop_back(), vals.drop_back()))) {
            Value keyVal = smt::IntConstantOp::create(
                builder, loc, builder.getI64IntegerAttr(key));
            Value thenVal = smt::IntConstantOp::create(
                builder, loc, builder.getI64IntegerAttr(val));
            Value cond = smt::EqOp::create(builder, loc, mappedIdx, keyVal);
            result = smt::IteOp::create(builder, loc, cond, thenVal, result);
          }
          mapping.map(lookupOp.getResult(), result);
        })
        .Default([&](Operation *bodyOp) { builder.clone(*bodyOp, mapping); });
  }

  smt::YieldOp::create(builder, loc);
  return solverOp;
}

// This function is exposed to the Python bindings.
OwningOpRef<ModuleOp>
convertConstraintsToSMTModule(IREE::Codegen::ConstraintsOp op) {
  OwningOpRef<ModuleOp> tempModule = ModuleOp::create(op->getLoc());
  OpBuilder builder(tempModule->getBodyRegion());
  convertConstraintsToSMTSolver(op, builder);
  return tempModule;
}

namespace {

struct ConvertConstraintsToSMTPass final
    : impl::ConvertConstraintsToSMTPassBase<ConvertConstraintsToSMTPass> {
  void runOnOperation() override {
    IREE::Codegen::ConstraintsOp constraintsOp = getOperation();
    OpBuilder builder(constraintsOp);
    mlir::smt::SolverOp solverOp =
        convertConstraintsToSMTSolver(constraintsOp, builder);

    // Erase the constraints body and replace it with the solverOp, so it won't
    // accidentally bypass the lit test with constraints contents.
    Block &bodyBlock = constraintsOp.getBody().front();
    bodyBlock.clear();
    solverOp->moveBefore(&bodyBlock, bodyBlock.end());
  }
};

} // namespace
} // namespace mlir::iree_compiler
