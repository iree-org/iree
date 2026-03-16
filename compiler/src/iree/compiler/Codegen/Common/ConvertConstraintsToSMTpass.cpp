// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTCONSTRAINTSTOSMTPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"



OwningOpRef<smt::SolverOp> convertConstraintsToSMTSolver(IREE::Codegen::ConstraintsOp op) {
    Location loc = op.getLoc();
    MLIRContext *ctx = op->getContext();
    OpBuilder builder(ctx);  

    auto solverOp = smt::SolverOp::create(builder, loc, TypeRange{}, ValueRange{});

    Block *solverBody = &solverOp.getBodyRegion().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(solverBody);

    IRMapping mapping;
    Block &constraintsBody = op.getBody().front();
    for (auto [idx, blockArg] : llvm::enumerate(constraintsBody.getArguments())) {
    std::string name = "problem_dim_" + std::to_string(idx);
    auto declareFun = smt::DeclareFunOp::create(
        builder, loc, blockArg.getType(), builder.getStringAttr(name));
    mapping.map(blockArg, declareFun.getResult());
    }

    for (Operation &bodyOp : constraintsBody) {
        TypeSwitch<Operation *>(&bodyOp)
            .Case<IREE::Codegen::AssertOp>([&](auto assertOp) {
                Value mappedCond = mapping.lookupOrDefault(assertOp.getCondition());
                smt::AssertOp::create(builder, loc, mappedCond);
            })
            .Case<IREE::Codegen::KnobOp>([&](auto knobOp) {
                auto declareFun = smt::DeclareFunOp::create(
                    builder, loc, knobOp.getResult().getType(), knobOp.getNameAttr());
                mapping.map(knobOp.getResult(), declareFun.getResult());
            })
            .Case<IREE::Codegen::LookupOp>([&](auto lookupOp) {
                Value mappedIdx = mapping.lookupOrDefault(lookupOp.getIndex());
                auto keys = lookupOp.getKeys();
                auto vals = lookupOp.getValues();

                Value result = smt::IntConstantOp::create(
                    builder, loc, builder.getIntegerAttr(builder.getIntegerType(64), vals.back()));

                for (auto [key, val] : llvm::reverse(llvm::zip(keys, vals))) {
                    Value keyVal = smt::IntConstantOp::create(
                        builder, loc, builder.getIntegerAttr(builder.getIntegerType(64), key));
                    Value thenVal = smt::IntConstantOp::create(
                        builder, loc, builder.getIntegerAttr(builder.getIntegerType(64), val));
                    Value cond = smt::EqOp::create(builder, loc, mappedIdx, keyVal);
                    result = smt::IteOp::create(builder, loc, cond, thenVal, result);
                }
                mapping.map(lookupOp.getResult(), result);
            })
            .Default([&](Operation *bodyOp) { builder.clone(*bodyOp, mapping); });
        
    }
    return solverOp;
}

namespace {

struct ConvertConstraintsToSMTPass final : impl::ConvertConstraintsToSMTPassBase<ConvertConstraintsToSMTPass> {
    void runOnOperation() override {
        auto constraintsOp = getOperation();
        OwningOpRef<smt::SolverOp> solverOp = convertConstraintsToSMTSolver(constraintsOp);

        // Swap the body ops for convertedSMT equivalents.
        // Block args are preserved to match the verifier's arg count check.
        Region &body = constraintsOp.getBody();
        body.front().clear();
        body.front().getOperations().splice(
            body.front().end(),
            solverOp->getBodyRegion().front().getOperations());
    }
};

} // namespace
} // namespace mlir::iree_compiler