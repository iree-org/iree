//====- LowerGemminiPass.cpp - Gemmini Dialect Lowering Pass  -------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines Gemmini dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"

using namespace mlir;
using namespace buddy;

// PrintOpLowering refers to the toy.print op.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(gemmini::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = rewriter.getContext();
    auto memRefType = llvm::cast<MemRefType>(*op->operand_type_begin());
    auto memRefShape = memRefType.getShape();
    Type memElementType = memRefType.getElementType();
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst;
    if (memElementType == rewriter.getF32Type() ||
        memElementType == rewriter.getF64Type()) {
      formatSpecifierCst = getOrCreateGlobalString(
          loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
    } else if (memElementType == rewriter.getI8Type() ||
               memElementType == rewriter.getI32Type()) {
      formatSpecifierCst = getOrCreateGlobalString(
          loc, rewriter, "frmt_spec", StringRef("%d \0", 4), parentModule);
    }
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      rewriter.setInsertionPointToEnd(loop.getBody());

      if (i != e - 1)
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                      newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto printOp = cast<gemmini::PrintOp>(op);
    Value elementLoad =
        rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    if (elementLoad.getType() == rewriter.getF32Type())
      elementLoad = rewriter.create<mlir::LLVM::FPExtOp>(
          loc, rewriter.getF64Type(), elementLoad);
    else if (elementLoad.getType() == rewriter.getI8Type())
      elementLoad = rewriter.create<mlir::LLVM::SExtOp>(
          loc, rewriter.getI32Type(), elementLoad);
    rewriter.create<LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    rewriter.eraseOp(op);
    return success();
  }

private:
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtr = LLVM::LLVMPointerType::get(context);
    return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtr, true);
  }

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }

  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value), 0);
    }

    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

namespace {
class LowerGemminiToLLVMPass
    : public PassWrapper<LowerGemminiToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemminiToLLVMPass)
  StringRef getArgument() const final { return "lower-gemmini"; }
  StringRef getDescription() const final {
    return "gemmini dialect lowering pass.";
  }
  LowerGemminiToLLVMPass() = default;
  LowerGemminiToLLVMPass(const LowerGemminiToLLVMPass &) {}

  Option<int64_t> dim{*this, "dim", llvm::cl::desc("Size of systolic array."),
                      llvm::cl::init(16)};
  Option<int64_t> addrLen{*this, "addr_len",
                          llvm::cl::desc("The length of address."),
                          llvm::cl::init(32)};
  Option<int64_t> accRows{*this, "acc_rows", llvm::cl::desc("The row of acc."),
                          llvm::cl::init(1024)};
  Option<int64_t> bankRows{*this, "bank_rows",
                           llvm::cl::desc("The row of the bank."),
                           llvm::cl::init(4096)};
  Option<std::string> elemType{*this, "elem_t",
                               llvm::cl::desc("The type of elem_t."),
                               llvm::cl::init("i8")};
  Option<std::string> accType{*this, "acc_t",
                              llvm::cl::desc("The type of acc_t."),
                              llvm::cl::init("i32")};

  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<gemmini::GemminiDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerGemminiToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  // The default elem_t is int8_t,
  // so the default size of elem_t is 1 type.
  size_t sizeOfElemT = sizeof(int8_t);
  if (elemType == "f32")
    sizeOfElemT = sizeof(float);
  // The default acc_t is int32_t,
  // so the default size of acc_t is 4 type.
  size_t sizeOfAccT = sizeof(int32_t);
  if (accType == "f32")
    sizeOfAccT = sizeof(float);
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  LLVMConversionTarget target(*context);
  configureGemminiLegalizeForExportTarget(target);
  populateGemminiLegalizeForLLVMExportPatterns(converter, patterns, dim,
                                               addrLen, accRows, bankRows,
                                               sizeOfElemT, sizeOfAccT);
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  patterns.add<PrintOpLowering>(&getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerGemminiPass() { PassRegistration<LowerGemminiToLLVMPass>(); }
} // namespace buddy
} // namespace mlir
