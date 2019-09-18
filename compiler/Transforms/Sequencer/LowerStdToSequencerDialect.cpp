// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/llvm/llvm/include/llvm/ADT/ArrayRef.h"
#include "third_party/llvm/llvm/include/llvm/ADT/DenseMap.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Dialect/StandardOps/Ops.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Attributes.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/BlockAndValueMapping.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Location.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/OperationSupport.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/StandardTypes.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/PassRegistry.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Support/LogicalResult.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Transforms/DialectConversion.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Transforms/Utils.h"
#include "third_party/mlir_edge/iree/compiler/IR/Dialect.h"
#include "third_party/mlir_edge/iree/compiler/IR/Ops.h"
#include "third_party/mlir_edge/iree/compiler/IR/Sequencer/HLDialect.h"
#include "third_party/mlir_edge/iree/compiler/IR/Sequencer/HLOps.h"
#include "third_party/mlir_edge/iree/compiler/IR/StructureOps.h"
#include "third_party/mlir_edge/iree/compiler/Utils/MemRefUtils.h"

namespace mlir {
namespace iree_compiler {

namespace {

class SequencerConversionPattern : public ConversionPattern {
 public:
  SequencerConversionPattern(StringRef operationName, int benefit,
                             MLIRContext *context,
                             MemRefTypeConverter &typeConverter)
      : ConversionPattern(operationName, benefit, context),
        typeConverter_(typeConverter) {}

 protected:
  MemRefTypeConverter &typeConverter_;
};

struct ConstantOpLowering : public SequencerConversionPattern {
  ConstantOpLowering(MLIRContext *context, MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(ConstantOp::getOperationName(), 1, context,
                                   typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    const auto &valueAttr = cast<ConstantOp>(op).getValue();
    auto midOp = rewriter.create<IREE::ConstantOp>(op->getLoc(), valueAttr);

    auto result = wrapAsTensor(midOp.getResult(), op, rewriter);
    rewriter.replaceOp(
        op, {loadResultValue(op->getLoc(), op->getResult(0)->getType(), result,
                             rewriter)});
    return matchSuccess();
  }
};

class CallOpLowering : public SequencerConversionPattern {
 public:
  CallOpLowering(MLIRContext *context, MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(CallOp::getOperationName(), 1, context,
                                   typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto callOp = cast<CallOp>(op);

    SmallVector<Type, 4> convertedResults;
    auto result = typeConverter_.convertTypes(
        callOp.getCalleeType().getResults(), convertedResults);
    (void)result;
    assert(succeeded(result) && "expected valid callee type conversion");
    rewriter.replaceOpWithNewOp<IREESeq::HL::CallOp>(
        op, callOp.getCallee(), convertedResults, operands);

    return matchSuccess();
  }
};

class CallIndirectOpLowering : public SequencerConversionPattern {
 public:
  CallIndirectOpLowering(MLIRContext *context,
                         MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(CallIndirectOp::getOperationName(), 1,
                                   context, typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto callOp = cast<CallIndirectOp>(op);
    rewriter.replaceOpWithNewOp<IREESeq::HL::CallIndirectOp>(
        op, callOp.getCallee(), operands);
    return matchSuccess();
  }
};

struct ReturnOpLowering : public SequencerConversionPattern {
  ReturnOpLowering(MLIRContext *context, MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(ReturnOp::getOperationName(), 1, context,
                                   typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value *, 4> newOperands;
    newOperands.reserve(operands.size());
    for (auto *operand : operands) {
      newOperands.push_back(wrapAsMemRef(operand, op, rewriter));
    }
    rewriter.replaceOpWithNewOp<IREESeq::HL::ReturnOp>(op, newOperands);
    return matchSuccess();
  }
};

struct BranchOpLowering : public SequencerConversionPattern {
  BranchOpLowering(MLIRContext *context, MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(BranchOp::getOperationName(), 1, context,
                                   typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREESeq::HL::BranchOp>(op, destinations[0],
                                                       operands[0]);
    return this->matchSuccess();
  }
};

struct CondBranchOpLowering : public SequencerConversionPattern {
  CondBranchOpLowering(MLIRContext *context, MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(CondBranchOp::getOperationName(), 1, context,
                                   typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *condValue =
        loadAccessValue(op->getLoc(), properOperands[0], rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::HL::CondBranchOp>(
        op, condValue, destinations[IREESeq::HL::CondBranchOp::trueIndex],
        operands[IREESeq::HL::CondBranchOp::trueIndex],
        destinations[IREESeq::HL::CondBranchOp::falseIndex],
        operands[IREESeq::HL::CondBranchOp::falseIndex]);
    return this->matchSuccess();
  }
};

class AllocOpLowering : public SequencerConversionPattern {
 public:
  AllocOpLowering(MLIRContext *context, MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(AllocOp::getOperationName(), 1, context,
                                   typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): replace with length computation.
    rewriter.replaceOpWithNewOp<IREESeq::HL::AllocHeapOp>(
        op, *op->getResultTypes().begin(), operands);
    return matchSuccess();
  }
};

class DeallocOpLowering : public SequencerConversionPattern {
 public:
  DeallocOpLowering(MLIRContext *context, MemRefTypeConverter &typeConverter)
      : SequencerConversionPattern(DeallocOp::getOperationName(), 1, context,
                                   typeConverter) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREESeq::HL::DiscardOp>(op, operands[0]);
    return matchSuccess();
  }
};

void populateStdToSequencerConversionPatterns(
    MLIRContext *context, MemRefTypeConverter &converter,
    OwningRewritePatternList &patterns) {
  patterns.insert<ConstantOpLowering,
                  // Control flow.
                  CallOpLowering, CallIndirectOpLowering, ReturnOpLowering,
                  BranchOpLowering, CondBranchOpLowering,
                  // Memory management.
                  AllocOpLowering, DeallocOpLowering>(context, converter);
}

}  // namespace

// Lowers functions using std.* ops to the IREE HL sequencer dialect and buffer
// view types.
// FuncOp signatures will be updated to use the buffer view type and
// dispatch regions will get iree.bind_input where needed.
//
// Beyond bindings there will be no other changes within dispatchable regions.
// It is up to the downstream dialects to properly use the bindings to map their
// I/O to expected values.
//
// Note that output buffer allocation is required following this pass to either
// elide dispatch results entirely and provide output params or provide both
// while ensuring that the returned value is always sliced from an input. This
// should happen prior to outlining.
class LowerStdToSequencerDialectPass
    : public ModulePass<LowerStdToSequencerDialectPass> {
 public:
  void runOnModule() override {
    auto module = getModule();

    // Only convert top-level functions, not ones nested in executables.
    std::vector<Operation *> toConvert;
    for (auto funcOp : module.getOps<FuncOp>()) {
      toConvert.push_back(funcOp);
    }

    // Convert the signature and body of all sequencer functions.
    MemRefTypeConverter converter(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<IREEHLSequencerDialect, IREEDialect>();
    target.addLegalOp<LoadOp, StoreOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    OwningRewritePatternList patterns;
    populateStdToSequencerConversionPatterns(&getContext(), converter,
                                             patterns);
    if (failed(
            applyPartialConversion(toConvert, target, patterns, &converter))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createLowerStdToSequencerDialectPass() {
  return std::make_unique<LowerStdToSequencerDialectPass>();
}

static PassRegistration<LowerStdToSequencerDialectPass> pass(
    "iree-lower-std-to-sequencer-dialect",
    "Lowers std ops to the IREE HL sequencer dialect.");

}  // namespace iree_compiler
}  // namespace mlir
