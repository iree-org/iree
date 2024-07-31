// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-topk-lowering"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define VEC_LDBG(X) LLVM_DEBUG(VEC_DBGS() << X << "\n")

namespace mlir::iree_compiler {

namespace {
class TopkLoweringPass : public TopkLoweringBase<TopkLoweringPass> {
public:
  using TopkLoweringBase::TopkLoweringBase;
  TopkLoweringPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::LinalgExt::IREELinalgExtDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

static LogicalResult
lowerTopkOpPreconditionUsingSCF(IREE::LinalgExt::TopkOp topkOp) {
  if (isTopkSCFLowerEnabled(topkOp))
    return success();
  return failure();
}

static scf::ForOp replaceForOpWithNewSignature(RewriterBase &rewriter,
                                               scf::ForOp loop,
                                               ValueRange newInitArgs) {
  OpBuilder::InsertionGuard g(rewriter);
  // Create a new loop before the existing one, with the extra operands.
  rewriter.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getInitArgs());
  llvm::append_range(operands, newInitArgs);
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      operands);
  rewriter.eraseBlock(newLoop.getBody());

  newLoop.getRegion().getBlocks().splice(
      newLoop.getRegion().getBlocks().begin(), loop.getRegion().getBlocks());

  for (Value operand : newInitArgs) {
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());
  }

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    rewriter.replaceAllUsesWith(std::get<0>(it), std::get<1>(it));

  LLVM_DEBUG(VEC_DBGS() << "newLoop now: " << newLoop << "\n");
  LLVM_DEBUG(VEC_DBGS() << "stripped scf.for: " << loop << "\n");
  LLVM_DEBUG(VEC_DBGS() << "erase: " << loop);

  rewriter.eraseOp(loop);
  return newLoop;
}

/// Add the necessary IterArgs to the input iterating loop.
// This function rewrites the outer loop of the vector tile and
// insert new ioter_args for the loop to support the topk
// algorithm implementation.
// Note: The iter_args to the loop is in the format:
// 1. Value for the out0 tensor (already added by the tiling)
// 2. Value for the out1 tensor (already added by the tiling)
// 3. an i1 value denoting wheter the smallestElem was initialized.
// 4. Value for the smallestElem (smallest element added to the out0 tensor)
// 5. Number of elements added to the out tensors.
static LogicalResult addIterArgs(RewriterBase &rewriter, Location loc,
                                 scf::ForOp loop, Type outValType,
                                 Type outIdxType) {
  llvm::DenseMap<Value, Value> valueMapping;
  SmallVector<Value> newOperands;
  SmallVector<std::pair<size_t, size_t>> argMapping;
  // First, copy the existing loop args.
  for (const auto &operand : llvm::enumerate(loop.getInitArgs())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end()) {
      LLVM_DEBUG(VEC_DBGS()
                 << "no value mapping for: " << operand.value() << "\n");
      continue;
    }

    argMapping.push_back(std::make_pair(
        operand.index(), loop.getInitArgs().size() + newOperands.size()));
    newOperands.push_back(it->second);
  }

  // Add an arg wheter smallestElem was initialized
  Value firstElemInit = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  newOperands.push_back(firstElemInit);

  // Add args placeholders for the smallest element added to the out array
  Value smallestOut;
  if (auto floatType = llvm::dyn_cast_if_present<FloatType>(outValType)) {
    smallestOut = rewriter.create<arith::ConstantFloatOp>(
        loc, mlir::APFloat(floatType.getFloatSemantics(), 0), floatType);
  } else if (auto intType =
                 llvm::dyn_cast_if_present<IntegerType>(outValType)) {
    smallestOut = rewriter.create<arith::ConstantIntOp>(loc, 0, intType);
  } else {
    // Unexpected type!
    return failure();
  }
  newOperands.push_back(smallestOut);

  // Add an arg for the number of elements added to the out array(s).
  Value numElemsAdded = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  newOperands.push_back(numElemsAdded);
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(rewriter, loop, newOperands);
  Block &loopBody = *newForOp.getBody();
  for (auto mapping : argMapping) {
    valueMapping[newForOp.getResult(mapping.first)] =
        newForOp.getResult(mapping.second);
    valueMapping[loopBody.getArgument(mapping.first +
                                      newForOp.getNumInductionVars())] =
        loopBody.getArgument(mapping.second + newForOp.getNumInductionVars());
  }
  return success();
}

struct InsertElemContinuationValues {
  Value out0;         // OutputValues
  Value out1;         // OutputIndices
  Value smallestElem; // Smallest Element
  Value addedElems;   // The number of elements added to the output
};

/// Insert a new, bigger element than the currently smallest element added in
/// output of the function.
/// This code is executed after a determination is made that an element from
/// the input will be included in the output. If no need to include an element
/// in the output, this code is never executed, thus making the building of the
/// output as close as possible to O(n).
///
/// 1. Find the position of the element that needs to be inserted.
///    There is no support for returning in the middle of a loop,
///    so an iter_arg flag is set to annotate that the insertion
///    index is found.
/// 2. If the insertion index is equal of the dimension of output,
///    no addition of elements is needed - just shift and replace.
/// 3. If the insertion index is equal of the elements added to the output,
///    No need to shift, just add the element at the end and expand the
///    addedElems.
/// Returns the out0, out1, smallestElem, addedElems values for continuation.
InsertElemContinuationValues
insertElemInOutput(Location loc, OpBuilder b, Value elemToInsertIfNeeded,
                   Value elemIndex, Value outputNum,
                   InsertElemContinuationValues continuation) {
  SmallVector<Value> newOperands;
  // Flag wheter we have not found the first smaller element.
  newOperands.push_back(b.create<arith::ConstantIntOp>(loc, 1, 1));
  // Keeps the index of where to insert the element that we return.
  newOperands.push_back(b.create<arith::ConstantIndexOp>(loc, 0));

  scf::YieldOp scfYieldOp;
  // The following loop finds the insertion index of the input element in
  // thr output.
  scf::ForOp findSmaller = b.create<scf::ForOp>(
      loc, b.create<arith::ConstantIndexOp>(loc, 0), continuation.addedElems,
      b.create<arith::ConstantIndexOp>(loc, 1), newOperands,
      [&](OpBuilder &bIn, Location nestedLoc, Value iv, ValueRange args) {
        scfYieldOp = b.create<scf::YieldOp>(nestedLoc, newOperands);
      });

  // Fill in the body for findSmaller Loop
  // We need the iter_args and they are not created yet in the lambda.
  {
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPoint(scfYieldOp);
    // Flag wheter we found the insertion point.
    Value notFoundSmaller = findSmaller.getRegionIterArgs()[0];
    // The index where the element needs to be inserted.
    Value insertionIndex = findSmaller.getRegionIterArgs()[1];
    // Test the 0th iter_arg to decide if the insertion index in the output
    // is found.
    // If we have found the insertionIndex in previous iterations, do nothing.
    // The insertion index is the index of the first element in out0 that is
    // smaller than the current element to be inserted, or the addedElems
    // (elements added to the output), if not the whole output is filled,
    // and there is no element smaller that the incoming element.
    // TODO: lubol this scf::If is needed because scf::ForOp has no support
    //       for break in a middle of a loop. The code will be simplified
    //       quite a bit (and be faster) when such support is added.
    //       Remove it when such support is added.
    Value cmpElemOp = b.create<arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, notFoundSmaller,
        b.create<arith::ConstantIntOp>(loc, 1, 1));
    auto ifFoundIndex = b.create<scf::IfOp>(
        loc, cmpElemOp,
        [&](OpBuilder &b, Location loc) {
          // Not found yet.
          Value notFoundSm = notFoundSmaller;
          Value insertionInd = insertionIndex;

          Value idxDim0 = b.create<arith::ConstantIndexOp>(loc, 0);
          Value inElem = b.create<tensor::ExtractOp>(
              loc, continuation.out0,
              ValueRange{idxDim0, findSmaller.getInductionVar()});
          // First, check if this is the position where the new element needs to
          // be inserted.
          if (auto floatType = llvm::dyn_cast_if_present<FloatType>(
                  elemToInsertIfNeeded.getType())) {
            cmpElemOp =
                b.create<arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT,
                                        inElem, elemToInsertIfNeeded);
          } else if (auto intType = llvm::dyn_cast_if_present<IntegerType>(
                         elemToInsertIfNeeded.getType())) {
            cmpElemOp =
                b.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                        inElem, elemToInsertIfNeeded);
          } else {
            assert(false && "Invalid type for topk vectorization!");
          }

          cmpElemOp = b.create<arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::eq, cmpElemOp,
              b.create<arith::ConstantIntOp>(loc, 1, 1));
          auto ifToAdd = b.create<scf::IfOp>(
              loc, cmpElemOp,
              [&](OpBuilder &b, Location loc) {
                // Get the value of the current induction variable,
                // which is the index where the new element needs to
                // be inserted in the output.
                Value insertionIndex = b.create<arith::AddIOp>(
                    loc, findSmaller.getInductionVar(),
                    b.create<arith::ConstantIndexOp>(loc, 0));
                SmallVector<Value> newOperands;
                // Set the flag that we have "found" the index where to insert.
                newOperands.push_back(
                    b.create<arith::ConstantIntOp>(loc, 0, 1));
                newOperands.push_back(insertionIndex);
                b.create<scf::YieldOp>(loc, newOperands);
              },
              [&](OpBuilder &b, Location loc) {
                // The index was not found yet.
                // Continue the iteration, not modifying the
                // returns of the if.
                Value notFoundSmIn = notFoundSm;
                Value insertionIndIn = insertionInd;
                SmallVector<Value> newOperands;
                newOperands.push_back(notFoundSmIn);
                newOperands.push_back(insertionIndIn);
                b.create<scf::YieldOp>(loc, newOperands);
              });

          SmallVector<Value> newOperands;
          // Return the current insertion index and found status (the one we
          // found in the if or the one that came in the if,
          // but was not updated).
          newOperands.push_back(ifToAdd.getResult(0));
          newOperands.push_back(ifToAdd.getResult(1));
          b.create<scf::YieldOp>(loc, newOperands);
        },
        [&](OpBuilder &b, Location loc) {
          // The index was already found yet.
          // Just return. No need to update anything.
          // Note: We still need to iterate to the end of the loop since
          // there is no break out of scf::ForOp.
          SmallVector<Value> newOperands;
          Value notFoundSm = notFoundSmaller;
          Value insertionInd = insertionIndex;
          newOperands.push_back(notFoundSm);
          newOperands.push_back(insertionInd);
          b.create<scf::YieldOp>(loc, newOperands);
        });

    notFoundSmaller = ifFoundIndex.getResult(0);
    insertionIndex = ifFoundIndex.getResult(1);

    SmallVector<Value> newOperands;
    // The for loop returns if insertion index was found
    // and the insertion index value.
    newOperands.push_back(notFoundSmaller);
    newOperands.push_back(insertionIndex);
    b.create<scf::YieldOp>(loc, newOperands);
    scfYieldOp.erase();
  }

  Value insertionIndexNotFound = findSmaller.getResult(0);
  Value insertionIndex = findSmaller.getResult(1);

  // If the insertionIndex is not found and the addedElems == output dimension,
  // do nothing. The element should not be added.
  Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                      continuation.addedElems, outputNum);
  cmp = b.create<arith::AndIOp>(loc, cmp, insertionIndexNotFound);
  auto ifAddElem = b.create<scf::IfOp>(
      loc, cmp,
      [&](OpBuilder &b, Location loc) {
        // if
        // Element doesn't need to be added to the output.
        // It is smaller than the last element in the out tensor.
        SmallVector<Value> newOperands;
        newOperands.push_back(continuation.out0);
        newOperands.push_back(continuation.out1);
        newOperands.push_back(continuation.smallestElem);
        newOperands.push_back(continuation.addedElems);
        b.create<scf::YieldOp>(loc, newOperands);
      },
      [&](OpBuilder &b, Location loc) {
        // else
        // If the insertionIndex not found, element is added at the end.
        // No need to shift.
        // The condition above exclude the case where we have filled the output
        // tensor already.
        auto ifAddAtEnd = b.create<scf::IfOp>(
            loc, insertionIndexNotFound,
            [&](OpBuilder &b, Location loc) {
              // if
              // Adding at addedElems and expanding the addedElems
              Value outValsIf = continuation.out0;
              Value outIndsIf = continuation.out1;
              ;
              Value smElemIf = continuation.smallestElem;
              Value addElemsIf = continuation.addedElems;
              Value idx0 = b.create<arith::ConstantIndexOp>(loc, 0);

              outValsIf = b.create<tensor::InsertOp>(
                  loc, elemToInsertIfNeeded, outValsIf,
                  ValueRange{idx0, addElemsIf});
              Value elemIndexI32 =
                  b.create<arith::IndexCastOp>(loc, b.getI32Type(), elemIndex);
              outIndsIf = b.create<tensor::InsertOp>(
                  loc, elemIndexI32, outIndsIf, ValueRange{idx0, addElemsIf});
              auto one = b.create<arith::ConstantIndexOp>(loc, 1);
              addElemsIf = b.create<arith::AddIOp>(loc, addElemsIf, one);
              SmallVector<Value> newOperandsIf;
              newOperandsIf.push_back(outValsIf);
              newOperandsIf.push_back(outIndsIf);
              newOperandsIf.push_back(smElemIf);
              newOperandsIf.push_back(addElemsIf);
              b.create<scf::YieldOp>(loc, newOperandsIf);
            },
            [&](OpBuilder &b, Location loc) {
              // The element needs to be inserted in the output tensor.
              // The elements in the output need to be shifted and the
              // new element to be set at the right index.
              Value outValsElse = continuation.out0;
              Value outIndsElse = continuation.out1;
              Value smElemElse = continuation.smallestElem;
              Value addElemsElse = continuation.addedElems;
              // Get the new end-index for shiftin the outputarray.
              // It is one less the last index of an added element in the
              // output, because the move is done using a[i+1] = a[i] shifting.
              Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  addElemsElse, outputNum);
              auto ifExpand = b.create<scf::IfOp>(
                  loc, cmp,
                  [&](OpBuilder &b, Location loc) {
                    // if case
                    // The the output is fully filled,
                    // no expansion is needed. Shift elements out of the output.
                    Value addElems = addElemsElse;
                    auto one = b.create<arith::ConstantIndexOp>(loc, 1);
                    Value lastElemIndex =
                        b.create<arith::SubIOp>(loc, addElems, one);
                    SmallVector<Value> newOperands;
                    newOperands.push_back(lastElemIndex);
                    b.create<scf::YieldOp>(loc, newOperands);
                  },
                  [&](OpBuilder &b, Location loc) {
                    // else
                    // Expansion is needed since the element is inserted
                    // and the elements after are shifted to eventually
                    // fill fully the output array (1-D tensor).
                    Value addElems = addElemsElse;
                    SmallVector<Value> newOperands;
                    newOperands.push_back(addElems);
                    b.create<scf::YieldOp>(loc, newOperands);
                  });

              Value lastElemIndex = ifExpand.getResult(0);
              // Replacing out[i+1] with out[i].
              // Make sure not over the index.
              auto one = b.create<arith::ConstantIndexOp>(loc, 1);
              // The following loop assumes that the iter_args are
              // the value of the element we are replacing in the next element,
              // for the both out arrays as 0th and 1st RegionIterArg.
              // Note: The elements of the a[i+1] are extracted, then the a[i+1]
              // elements are replaced with a[i] elements, and then the
              // extracted a[i+1] are placed as iter_args 0 and 1.
              Value idx0 = b.create<arith::ConstantIndexOp>(loc, 0);
              Value out1Repl = b.create<tensor::ExtractOp>(
                  loc, outValsElse, ValueRange{idx0, insertionIndex});
              Value out2Repl = b.create<tensor::ExtractOp>(
                  loc, outIndsElse, ValueRange{idx0, insertionIndex});
              SmallVector<Value> newOperands;
              newOperands.push_back(outValsElse);
              newOperands.push_back(outIndsElse);
              newOperands.push_back(out1Repl);
              newOperands.push_back(out2Repl);
              scf::YieldOp lastLoopOp;
              // Now shift the elements
              // This loop shift elements.
              // The logic is rather complicated because the scf::ForOp
              // doesn't support negative step -
              // https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop
              // The element values (for out0 and out1) that needs to be shifted
              // to the next element are passed as iter_args and thus avoiding a
              // temp var assignments.
              //
              // TODO: lubol remove the iter_args for the value that need to be
              // shifted to the next element when scf::ForOp support for
              // negative loop step is added.
              scf::ForOp shiftElemsLoop = b.create<scf::ForOp>(
                  loc, insertionIndex, lastElemIndex,
                  b.create<arith::ConstantIndexOp>(loc, 1), newOperands,
                  [&](OpBuilder &bIn, Location nestedLoc, Value iv,
                      ValueRange args) {
                    lastLoopOp =
                        bIn.create<scf::YieldOp>(nestedLoc, newOperands);
                  });

              // Create the loop body.
              // Need access to RegionIterArgs, which are not constructed in the
              // lambda yet.
              {
                PatternRewriter::InsertionGuard guard(b);
                // auto ip = rewriter.saveInsertionPoint();
                b.setInsertionPoint(lastLoopOp);
                Value nextIndVar = shiftElemsLoop.getInductionVar();
                auto one = b.create<arith::ConstantIndexOp>(loc, 1);
                nextIndVar = b.create<arith::AddIOp>(loc, nextIndVar, one);
                Value idx0 = b.create<arith::ConstantIndexOp>(loc, 0);
                Value out1Repl = b.create<tensor::ExtractOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[0],
                    ValueRange{idx0, nextIndVar});
                Value out2Repl = b.create<tensor::ExtractOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[1],
                    ValueRange{idx0, nextIndVar});
                Value newVals = b.create<tensor::InsertOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[2],
                    shiftElemsLoop.getRegionIterArgs()[0],
                    ValueRange{idx0, nextIndVar});
                Value newInds = b.create<tensor::InsertOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[3],
                    shiftElemsLoop.getRegionIterArgs()[1],
                    ValueRange{idx0, nextIndVar});
                SmallVector<Value> newOperands;
                newOperands.push_back(newVals);
                newOperands.push_back(newInds);
                newOperands.push_back(out1Repl);
                newOperands.push_back(out2Repl);
                b.create<scf::YieldOp>(loc, newOperands);
              }

              lastLoopOp.erase();

              // Get the smellest element as the last added element
              // to return to the next iteration of the tile (1x16xf/i32) loop.
              smElemElse = b.create<tensor::ExtractOp>(
                  loc, shiftElemsLoop.getResult(0),
                  ValueRange{idx0, ifExpand.getResult(0)});
              addElemsElse =
                  b.create<arith::AddIOp>(loc, ifExpand.getResult(0), one);
              outValsElse = shiftElemsLoop.getResult(0);
              outIndsElse = shiftElemsLoop.getResult(1);

              // Insert the element in the output.
              outValsElse = b.create<tensor::InsertOp>(
                  loc, elemToInsertIfNeeded, outValsElse,
                  ValueRange{idx0, insertionIndex});
              outIndsElse = b.create<tensor::InsertOp>(
                  loc,
                  b.create<arith::IndexCastOp>(loc, b.getIntegerType(32),
                                               elemIndex),
                  outIndsElse, ValueRange{idx0, insertionIndex});
              SmallVector<Value> newOperandsElse;
              newOperandsElse.push_back(outValsElse);
              newOperandsElse.push_back(outIndsElse);
              newOperandsElse.push_back(smElemElse);
              newOperandsElse.push_back(addElemsElse);
              b.create<scf::YieldOp>(loc, newOperandsElse);
            });

        // Propagate returns from nested loops and ifs to the
        // output of the function.
        Value outValsAtEnd = ifAddAtEnd.getResult(0);
        Value outIndsAtEnd = ifAddAtEnd.getResult(1);
        Value smElemAtEnd = ifAddAtEnd.getResult(2);
        Value addElemsAtEnd = ifAddAtEnd.getResult(3);

        SmallVector<Value> newOperands;
        newOperands.push_back(outValsAtEnd);
        newOperands.push_back(outIndsAtEnd);
        newOperands.push_back(smElemAtEnd);
        newOperands.push_back(addElemsAtEnd);
        b.create<scf::YieldOp>(loc, newOperands);
      });
  // Build and return the continuation of the outer loop,
  // after inserting the element.
  return InsertElemContinuationValues{
      ifAddElem.getResult(0), ifAddElem.getResult(1), ifAddElem.getResult(2),
      ifAddElem.getResult(3)};
}

/// Vectorize a `topKOp` with (1) static result and input types
static LogicalResult lowerAsLinalgExtTopkUsingSCF(
    RewriterBase &rewriter, IREE::LinalgExt::TopkOp topkOp,
    ArrayRef<int64_t> inputVectorSizes, SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(topkOp);
  Location loc = topkOp.getLoc();
  bool expectedRanksAndDims = true;
  ReifiedRankedShapedTypeDims reifiedReturnShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(topkOp.getOperation())
          .reifyResultShapes(rewriter, reifiedReturnShapes);
  assert(succeeded(status) && "failed to reify result shapes");

  if (failed(status))
    return failure();

  auto out0 = topkOp.getResults()[0];
  auto resShapedTy0 = llvm::cast<ShapedType>(out0.getType());
  auto out1 = topkOp.getResults()[1];
  auto resShapedTy1 = llvm::cast<ShapedType>(out1.getType());

  scf::ForOp scfInputLoop =
      dyn_cast<scf::ForOp>(topkOp->getParentRegion()->getParentOp());
  if (!scfInputLoop)
    return failure();

  auto outIdxElemType = resShapedTy1.getElementType();
  auto outValElemType = resShapedTy0.getElementType();

  {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(scfInputLoop);
    // Create a new loop based on the old one with adding the necessary
    // iter_args.
    // Note: The iter_args to the loop and all the high level scf::For
    // and scf::If are always in the same format:
    // 1. Value for the out0 tensor
    // 2. Value for the out1 tensor
    // 3. Value for the smallestElem (smallest element added to the out0 tensor)
    // 4. Number of elements added to the out tensors.
    // For the main loop there is another iter_arg:
    // 2.5. an i1 value denoting wheter the smallestElem was initialized.
    if (failed(addIterArgs(rewriter, loc, scfInputLoop, outValElemType,
                           outIdxElemType)))
      return failure();
  }

  // Get the newly created loop.
  auto regParent = topkOp->getParentRegion()->getParentOp();
  if (!isa<scf::ForOp>(regParent))
    return failure();
  scf::ForOp newLoop = dyn_cast<scf::ForOp>(regParent);
  Value outputInitialized = newLoop.getRegionIterArgs()[2];
  // The following if is used to initialize the smallestElem value.
  // It is done only once on the first iteration of the loop.
  // The outputs are the out0, out1, the smallest element added to the out0,
  // number of added elements.
  auto ifInitOut = rewriter.create<scf::IfOp>(
      loc, outputInitialized,
      [&](OpBuilder &b, Location loc) {
        Value outVals = newLoop.getRegionIterArgs()[0];
        // Out indexes
        Value outInds = newLoop.getRegionIterArgs()[1];
        Value idx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value firstInElem = rewriter.create<tensor::ExtractOp>(
            loc, topkOp.getValues(), ValueRange{idx0, idx0});
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        // The operands are:
        // 1. OutValues
        // 2. OutIndeces
        // 3. smallest added element.
        // 4. Number of added elements
        SmallVector<Value> operands;
        operands.push_back(outVals);
        operands.push_back(outInds);
        operands.push_back(firstInElem);
        operands.push_back(zero);
        b.create<scf::YieldOp>(loc, operands);
      },
      [&](OpBuilder &b, Location loc) {
        // It is initialize. Just return the init_args values.
        Value outVals = newLoop.getRegionIterArgs()[0];
        // Out indexes
        Value outInds = newLoop.getRegionIterArgs()[1];
        SmallVector<Value> operands;
        // If the smallest element has been initialized,
        // Don't do anything.
        operands.push_back(outVals);
        operands.push_back(outInds);
        operands.push_back(newLoop.getRegionIterArgs()[3]);
        operands.push_back(newLoop.getRegionIterArgs()[4]);
        b.create<scf::YieldOp>(loc, operands);
      });

  Type inElemTy = topkOp.getInputType().getElementType();
  TypedAttr minAttr;
  if (auto intType = llvm::dyn_cast<IntegerType>(inElemTy)) {
    minAttr = rewriter.getIntegerAttr(
        intType, APInt::getSignedMinValue(intType.getWidth()));
  } else if (auto floatType = llvm::dyn_cast<FloatType>(inElemTy)) {
    auto minApFloat =
        APFloat::getInf(llvm::cast<FloatType>(inElemTy).getFloatSemantics(),
                        /*Negative=*/true);
    minAttr = rewriter.getFloatAttr(inElemTy, minApFloat);
  } else {
    return failure();
  }

  Value padValue = rewriter.create<arith::ConstantOp>(loc, minAttr);
  Value inValsVec = vector::createReadOrMaskedRead(
      rewriter, loc, topkOp.getInputs()[0], inputVectorSizes, padValue, true);
  auto elemVecType = VectorType::get(inputVectorSizes, inElemTy);
  Value smallestMask = rewriter.create<vector::BroadcastOp>(
      loc, elemVecType, ifInitOut.getResult(2));
  Value comparedMask;
  if (auto floatType = llvm::dyn_cast_if_present<FloatType>(outValElemType)) {
    comparedMask = rewriter.create<arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, inValsVec, smallestMask);
  } else if (auto intType =
                 llvm::dyn_cast_if_present<IntegerType>(outValElemType)) {
    comparedMask = rewriter.create<arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, inValsVec, smallestMask);
  } else {
    return failure();
  }

  Value outNumElems =
      rewriter.create<arith::ConstantIndexOp>(loc, resShapedTy1.getDimSize(1));
  // If the output is not filled yet, set the masks to true.
  Value cmpAddedElems =
      rewriter.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                     ifInitOut.getResult(3), outNumElems);
  // This scf::If checks to see if the output is fully full.
  // If so, the mask we calculated is usable. If not, we should add as many
  // elements as needed in oder to fully fill the output.
  auto ifFillMask = rewriter.create<scf::IfOp>(
      loc, cmpAddedElems,
      [&](OpBuilder &b, Location loc) {
        Value tVal = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
        Value allElemsMask = rewriter.create<vector::BroadcastOp>(
            loc, comparedMask.getType(), tVal);
        SmallVector<Value> operands;
        operands.push_back(allElemsMask);
        b.create<scf::YieldOp>(loc, operands);
      },
      [&](OpBuilder &b, Location loc) {
        // else
        SmallVector<Value> operands;
        operands.push_back(comparedMask);
        b.create<scf::YieldOp>(loc, operands);
      });

  // Or the outs of the element compares to know if
  // there are any elements at all that need to be inserted.
  auto vecCmpCond = rewriter.create<vector::MultiDimReductionOp>(
      loc, ifFillMask.getResult(0),
      rewriter.create<arith::ConstantIntOp>(loc, 0, 1), SmallVector<bool>{1, 1},
      vector::CombiningKind::OR);

  // Check to see if any of the insertion logic needs to be triggered
  // (do we need to insert any elemets from the corrent ones
  // in the input vector).
  auto ifVecCmp = rewriter.create<scf::IfOp>(
      loc, vecCmpCond,
      [&](OpBuilder &b, Location loc) {
        // There are bigger elemnets than the smallest added,
        // so the insertion logic needs to be invoked.
        SmallVector<Value> maskProcessingLoopOps;
        maskProcessingLoopOps.push_back(ifInitOut.getResult(0));
        maskProcessingLoopOps.push_back(ifInitOut.getResult(1));
        maskProcessingLoopOps.push_back(ifInitOut.getResult(2));
        maskProcessingLoopOps.push_back(ifInitOut.getResult(3));
        scf::YieldOp forYield;
        // A loop that processes the masks in the vector register.
        // An mask of 1 denotes an element that needs to be inserted.
        scf::ForOp maskProcessingLoop = b.create<scf::ForOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, 0), newLoop.getStep(),
            b.create<arith::ConstantIndexOp>(loc, 1), maskProcessingLoopOps,
            [&](OpBuilder &bIn, Location nestedLoc, Value iv, ValueRange args) {
              forYield = bIn.create<scf::YieldOp>(loc);
            });

        {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPoint(forYield);
          SmallVector<OpFoldResult, 2> extractionIndices;
          extractionIndices.push_back(rewriter.getIndexAttr(0));
          extractionIndices.push_back(maskProcessingLoop.getInductionVar());
          auto sourceVectorType =
              dyn_cast<VectorType>(ifFillMask.getResult(0).getType());
          if (!sourceVectorType)
            expectedRanksAndDims = false;

          // The type here is always in the form 1x16xi1
          if (sourceVectorType.getRank() != 2)
            expectedRanksAndDims = false;

          bool hasLeadingDimUnitFixed =
              ((sourceVectorType.getShape().front() == 1) &&
               (!sourceVectorType.getScalableDims().front()));
          if (!hasLeadingDimUnitFixed)
            expectedRanksAndDims = false;
          VectorType newVType =
              VectorType::Builder(sourceVectorType).dropDim(0);

          // Drop leading/trailing unit dim by applying vector.shape_cast to all
          // operands
          Value aCast = rewriter.create<vector::ShapeCastOp>(
              loc, newVType, ifFillMask.getResult(0));
          // Ecxtarct from the input vector the element
          // that needs to be inserted.
          auto elemMask = rewriter.create<vector::ExtractElementOp>(
              loc, aCast, maskProcessingLoop.getInductionVar());
          auto insertElemIfNeeded = rewriter.create<scf::IfOp>(
              loc, elemMask,
              [&](OpBuilder &bIn2, Location loc) {
                // We need to insert the element in the output.
                auto sourceVectorType =
                    dyn_cast<VectorType>(inValsVec.getType());
                if (!sourceVectorType)
                  expectedRanksAndDims = false;
                // The type here is always in the form 1x16xf/i32
                // The ExtractElementOp (or ExtractOp) of vector doesn't
                // support the unit dim types, so the first dim needs to
                // be dropped.
                if (sourceVectorType.getRank() != 2)
                  expectedRanksAndDims = false;
                bool hasLeadingDimUnitFixed =
                    ((sourceVectorType.getShape().front() == 1) &&
                     (!sourceVectorType.getScalableDims().front()));
                if (!hasLeadingDimUnitFixed)
                  expectedRanksAndDims = false;
                VectorType newVType =
                    VectorType::Builder(sourceVectorType).dropDim(0);
                // Drop leading/trailing unit dim by applying vector.shape_cast
                // to all operands
                Value aCast =
                    bIn2.create<vector::ShapeCastOp>(loc, newVType, inValsVec);
                auto elemToInsertIfNeeded =
                    bIn2.create<vector::ExtractElementOp>(
                        loc, aCast, maskProcessingLoop.getInductionVar());
                // Add the indexes of the outer (tile) loop and the vector
                // element loop to get the index of the
                // element we are inserting.
                Value elemIndex = bIn2.create<arith::AddIOp>(
                    loc, maskProcessingLoop.getInductionVar(),
                    newLoop.getInductionVar());
                // Call the function that generates the element insertion
                // code in the output.
                InsertElemContinuationValues cont = insertElemInOutput(
                    loc, bIn2, elemToInsertIfNeeded, elemIndex, outNumElems,
                    InsertElemContinuationValues{
                        maskProcessingLoop.getRegionIterArgs()[0],
                        maskProcessingLoop.getRegionIterArgs()[1],
                        maskProcessingLoop.getRegionIterArgs()[2],
                        maskProcessingLoop.getRegionIterArgs()[3]});
                SmallVector<Value> operands;
                operands.push_back(cont.out0);
                operands.push_back(cont.out1);
                operands.push_back(cont.smallestElem);
                operands.push_back(cont.addedElems);
                bIn2.create<scf::YieldOp>(loc, operands);
              },
              [&](OpBuilder &bIn3, Location loc) {
                // else No need to insert anything.
                // Just return the tensor and the smallestElem,
                // and elemsAdded without changes.
                SmallVector<Value> operands;
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[0]);
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[1]);
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[2]);
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[3]);
                bIn3.create<scf::YieldOp>(loc, operands);
              });
          // Propagate outputs of ifs/fors.
          SmallVector<Value> newOperands;
          newOperands.push_back(insertElemIfNeeded.getResult(0));
          newOperands.push_back(insertElemIfNeeded.getResult(1));
          newOperands.push_back(insertElemIfNeeded.getResult(2));
          newOperands.push_back(insertElemIfNeeded.getResult(3));
          rewriter.create<scf::YieldOp>(loc, newOperands);
          forYield.erase();
        }
        SmallVector<Value> operands;
        operands.push_back(maskProcessingLoop.getResult(0));
        operands.push_back(maskProcessingLoop.getResult(1));
        operands.push_back(maskProcessingLoop.getResult(2));
        operands.push_back(maskProcessingLoop.getResult(3));
        b.create<scf::YieldOp>(loc, operands);
      },
      [&](OpBuilder &b, Location loc) {
        // else
        // No bigger elements. No work to do.
        SmallVector<Value> operands;
        operands.push_back(ifInitOut.getResult(0));
        operands.push_back(ifInitOut.getResult(1));
        operands.push_back(ifInitOut.getResult(2));
        operands.push_back(ifInitOut.getResult(3));
        b.create<scf::YieldOp>(loc, operands);
      });

  // If the types were not expected, don't do rewriting.
  if (!expectedRanksAndDims)
    return failure();

  // Create a new yield op for the top (tiling) loop.
  {
    OpBuilder::InsertionGuard g(rewriter);
    scf::YieldOp yieldOp =
        llvm::cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> operands;
    operands.push_back(ifVecCmp.getResult(0));
    operands.push_back(ifVecCmp.getResult(1));
    // After first run the out arrays are initialized.
    operands.push_back(
        rewriter.create<arith::ConstantIntOp>(yieldOp.getLoc(), 0, 1));
    operands.push_back(ifVecCmp.getResult(2));
    operands.push_back(ifVecCmp.getResult(3));
    rewriter.create<scf::YieldOp>(yieldOp.getLoc(), operands);
    rewriter.eraseOp(yieldOp);
  }
  newResults.push_back(topkOp.outputValues());
  newResults.push_back(topkOp.outputIndices());
  return success();
}

void TopkLoweringPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  IRRewriter rewriter(context);
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<IREE::LinalgExt::TopkOp>(op) &&
        isTopkSCFLowerEnabled(dyn_cast<IREE::LinalgExt::TopkOp>(op))) {
      candidates.push_back(op);
    }
  });

  // The vector input sizes inference needs to use producers, so we apply
  // vectorization from bottom to top.
  std::reverse(candidates.begin(), candidates.end());
  for (Operation *op : candidates) {
    if (auto topkOp = dyn_cast<IREE::LinalgExt::TopkOp>(op)) {
      auto arg0 = topkOp.getInputs()[0];
      auto shapedTy = llvm::cast<ShapedType>(arg0.getType());
      SmallVector<int64_t> vectorSizes;
      vectorSizes.append(shapedTy.getShape().begin(),
                         shapedTy.getShape().end());
      if (shapedTy.isDynamicShape(vectorSizes))
        continue;
      if (failed(lowerTopkOpPreconditionUsingSCF(topkOp))) {
        VEC_LDBG("Vectorization TopK pre-conditions failed\n");
        return; // falied.
      }
      SmallVector<Value> results;
      if (failed(lowerAsLinalgExtTopkUsingSCF(rewriter, topkOp, vectorSizes,
                                              results))) {
        VEC_LDBG("TopK Vectorization failed\n");
        return;
      }
      if (!results.empty())
        rewriter.replaceOp(op, results);
      else
        rewriter.eraseOp(op);
    }
  };

  {
    // Canonicalize mask related ops before we lower them.
    RewritePatternSet maskCanonPatterns(funcOp.getContext());
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                      funcOp.getContext());
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                        funcOp.getContext());
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                funcOp.getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(maskCanonPatterns)))) {
      return signalPassFailure();
    }
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTopkLoweringPass() {
  return std::make_unique<TopkLoweringPass>();
}

} // namespace mlir::iree_compiler
