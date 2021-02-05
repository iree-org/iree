// Copyright 2020 Google LLC
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

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

static constexpr int kIndexPackedBuffer = 0;
static constexpr int kIndexPushConstant = 1;
static constexpr int kIndexWorkGroupId = 2;
static constexpr int kIndexWorkGroupCount = 3;
static constexpr int kIndexWorkGroupSize = 4;

// Return the index type: i32.
static Type getIndexTy(MLIRContext *context) {
  return IntegerType::get(context, 32);
}

// func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
//          %push_constant: !llvm.ptr<i32>,
//          workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//          workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//          workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
static SmallVector<Type, 5> getABITypes(MLIRContext *context) {
  auto indexTy = getIndexTy(context);
  return SmallVector<Type, 5>{
      // %packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>
      LLVM::LLVMPointerType::get(
          LLVM::LLVMPointerType::get(IntegerType::get(context, 8))),
      // %push_constant: !llvm.ptr<i32>
      LLVM::LLVMPointerType::get(indexTy),
      // %workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>
      LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(indexTy, 3)),
      // %workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>
      LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(indexTy, 3)),
      // %workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>
      LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(indexTy, 3))};
}

// Convert to an LLVMFuncOp form with LLVM types. This implements an ABI that is
// compatible with IREE and which cannot be represented in std atm.
// Since it cannot be represented in std, this is an IREE-specific conversion.
// The conversion rewrites the argument-less and return-less function:
//
// ```
//    func @foo() { ... }
// ```
//
// into:
//
// ```
//    llvm.func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
//                  %push_constant: !llvm.ptr<i32>,
//                  workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
// ```
//
// Bump the benefit of the pattern to 100 to pick this pattern instead of a
// competing pattern inserted by `populateStdToLLVMConversionPatterns`.
class ConvertFunc : public ConvertToLLVMPattern {
 public:
  explicit ConvertFunc(MLIRContext *context, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(mlir::FuncOp::getOperationName(), context,
                             converter, 100) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);
    FunctionType fnType = funcOp.getType();
    (void)fnType;
    if (!funcOp.isPublic()) return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    // func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
    //          %push_constant: !llvm.ptr<i32>,
    //          workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
    //          workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
    //          workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    MLIRContext *context = rewriter.getContext();
    SmallVector<Type, 5> llvmInputTypes = getABITypes(context);
    signatureConverter.addInputs(llvmInputTypes);

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : funcOp.getAttrs()) {
      if (attr.first == SymbolTable::getSymbolAttrName() ||
          attr.first == mlir::impl::getTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), llvmInputTypes);
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmFuncType,
        LLVM::Linkage::External, funcAttrs);

    // Copy all of funcOp's operations into newFuncOp's body and perform region
    // type conversion.
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConverter)))
      return failure();

    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// Returns the IREE::HAL::InterfaceBindingOp from an interface op.
// TODO(ravishankarm) : Copy of similar method in LinalgToSPIRV path. Consider
// combining them.
IREE::HAL::InterfaceBindingOp getBindingOp(Operation *op) {
  if (auto placeholderOp = dyn_cast<IREE::PlaceholderOp>(op)) {
    return cast<IREE::HAL::InterfaceBindingOp>(
        SymbolTable::lookupNearestSymbolFrom(
            op, op->getAttrOfType<SymbolRefAttr>("binding")));
  }
  if (auto bindingSubspanOp =
          dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(op)) {
    return bindingSubspanOp.queryBindingOp();
  }
  llvm_unreachable("unknown interface binding op");
}

// Assumes the enclosing FuncOp has been converted to IREE-compatible ABI:
// ```
//    llvm.func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
//                  %push_constant: !llvm.ptr<i32>,
//                  workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
// ```
//
// Rewrites BindingOp into the proper memref descriptor extracted from
// `packed_buffer_args`.
template <typename BindingOp>
class ConvertBindingOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertBindingOp(MLIRContext *context, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(BindingOp::getOperationName(), context,
                             converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    assert(llvmFuncOp.getNumArguments() > 0);

    auto llvmTypeConverter = getTypeConverter();
    Location loc = op->getLoc();
    MemRefType memrefType = op->getResult(0).getType().dyn_cast<MemRefType>();
    auto elementType = typeConverter->convertType(memrefType.getElementType());

    // Fetch the interface binding op and extract the buffer index from void**.
    auto interfaceBindingOp = getBindingOp(op);
    Value bufferIndex =
        rewriter.create<ConstantIndexOp>(loc, interfaceBindingOp.binding());
    Value llvmBufferIndex = rewriter.create<LLVM::DialectCastOp>(
        loc, llvmTypeConverter->convertType(bufferIndex.getType()),
        bufferIndex);
    Value llvmBufferBasePtrAddr = rewriter.create<LLVM::GEPOp>(
        loc, llvmFuncOp.getArgument(kIndexPackedBuffer).getType(),
        llvmFuncOp.getArgument(kIndexPackedBuffer), llvmBufferIndex);

    Value packedBuffersArgsPtrCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(
            elementType, memrefType.getMemorySpace())),
        llvmBufferBasePtrAddr);

    Value llvmBufferBasePtr =
        rewriter.create<LLVM::LoadOp>(loc, packedBuffersArgsPtrCasted);
    if (memrefType.hasStaticShape()) {
      auto desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefType, llvmBufferBasePtr);
      rewriter.replaceOp(op, {desc});
    } else {
      auto desc = MemRefDescriptor::undef(
          rewriter, loc, typeConverter->convertType(memrefType));
      desc.setAllocatedPtr(rewriter, loc, llvmBufferBasePtr);
      desc.setAlignedPtr(rewriter, loc, llvmBufferBasePtr);
      rewriter.replaceOp(op, {desc});
    }

    return success();
  }
};

class RemoveMakeRankedShape : public ConvertToLLVMPattern {
 public:
  explicit RemoveMakeRankedShape(MLIRContext *context,
                                 LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(Shape::MakeRankedShapeOp::getOperationName(),
                             context, typeconverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Check users are ops are going to be folded away.
    for (auto user : op->getUsers()) {
      if (!cast<Shape::TieShapeOp>(user) && !cast<Shape::RankedDimOp>(user))
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Upateds memref descriptors shape and strides informations and fold tie_shape
// into updated memref descriptor.
class ConvertTieShapePattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertTieShapePattern(MLIRContext *context,
                                  LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(Shape::TieShapeOp::getOperationName(), context,
                             typeconverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto tieShapeOp = cast<Shape::TieShapeOp>(op);
    auto loc = tieShapeOp.getLoc();
    MemRefDescriptor sourceMemRef(operands.front());
    auto makeRankedShapeOp =
        cast<Shape::MakeRankedShapeOp>(tieShapeOp.shape().getDefiningOp());
    auto rankedShapeType = makeRankedShapeOp.shape()
                               .getType()
                               .dyn_cast_or_null<Shape::RankedShapeType>();
    if (!rankedShapeType) return failure();

    auto shape = rankedShapeType.getAllDims();

    // Update memref descriptor shape and strides.
    // dynamic dim maybe in mid location, like shapex.ranked_shape<[128,?,128]
    int dynIdx = 0;
    for (int i = 0; i < shape.size(); ++i) {
      if (shape[i] == ShapedType::kDynamicSize) {
        sourceMemRef.setSize(rewriter, loc, i,
                             makeRankedShapeOp.dynamic_dimensions()[dynIdx++]);
      } else {
        sourceMemRef.setConstantSize(rewriter, loc, i, shape[i]);
      }
    }
    // Compute and update memref descriptor strides. Assumption here is memrefs
    // are row-major e.g following index linearization x[i, j, k] = i * x.dim[1]
    // * x.dim[2] + j * x.dim[2] + k
    sourceMemRef.setConstantStride(rewriter, loc, shape.size() - 1, 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
      auto stride = sourceMemRef.stride(rewriter, loc, i + 1);
      auto dim = sourceMemRef.size(rewriter, loc, i + 1);
      Value strideVal = rewriter.create<LLVM::MulOp>(loc, stride, dim);
      sourceMemRef.setStride(rewriter, loc, i, strideVal);
    }
    rewriter.replaceOp(tieShapeOp, {sourceMemRef});
    return success();
  }
};  // namespace iree_compiler

class ConvertHALInterfaceLoadConstant : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceLoadConstant(MLIRContext *context,
                                           LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceLoadConstantOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();

    assert(llvmFuncOp.getNumArguments() > 0);

    auto llvmTypeConverter = getTypeConverter();
    Location loc = op->getLoc();
    auto interfaceLoadConstantOp = cast<IREE::HAL::InterfaceLoadConstantOp>(op);
    Value pushConstantOffset = rewriter.create<ConstantIndexOp>(
        loc, interfaceLoadConstantOp.offset().getSExtValue());
    Value llvmPushConstantOffset = rewriter.create<LLVM::DialectCastOp>(
        loc, llvmTypeConverter->convertType(pushConstantOffset.getType()),
        pushConstantOffset);
    Value llvmPushConstantOffsetAddr = rewriter.create<LLVM::GEPOp>(
        loc, llvmFuncOp.getArgument(kIndexPushConstant).getType(),
        llvmFuncOp.getArgument(kIndexPushConstant), llvmPushConstantOffset);
    Value llvmPushConstantValue =
        rewriter.create<LLVM::LoadOp>(loc, llvmPushConstantOffsetAddr);
    if (interfaceLoadConstantOp.result().getType().isIndex()) {
      llvmPushConstantValue = rewriter.create<LLVM::ZExtOp>(
          loc, llvmTypeConverter->convertType(rewriter.getIndexType()),
          llvmPushConstantValue);
    }
    rewriter.replaceOp(op, llvmPushConstantValue);
    return success();
  }
};

class RemoveInterfaceOpPattern : public ConvertToLLVMPattern {
 public:
  explicit RemoveInterfaceOpPattern(MLIRContext *context,
                                    LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(IREE::HAL::InterfaceOp::getOperationName(),
                             context, typeconverter) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// ArgIndex hardcodes knowledge of argument position of `Op` in the
// IREE-compatible ABI.
template <typename Op, int ArgIndex>
class ConvertWorkgroupInfoOpPattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertWorkgroupInfoOpPattern(MLIRContext *context,
                                         LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(Op::getOperationName(), context, converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    // Bail until enclosing FuncOp has been converted.
    if (!newFuncOp) return failure();

    Location loc = op->getLoc();
    auto xyzArrayPtr = newFuncOp.getArgument(ArgIndex);
    auto xyzArrayValue = rewriter.createOrFold<LLVM::LoadOp>(loc, xyzArrayPtr);
    auto dimValue = rewriter.createOrFold<LLVM::ExtractValueOp>(
        loc, getIndexTy(op->getContext()), xyzArrayValue,
        rewriter.getI32ArrayAttr(ArrayRef<int>(
            op->getAttrOfType<IntegerAttr>("dimension").getInt())));

    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(
        op, typeConverter->convertType(op->getResult(0).getType()), dimValue);
    return success();
  }
};

struct ConvertToLLVMPass
    : public PassWrapper<ConvertToLLVMPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

}  // namespace

void ConvertToLLVMPass::runOnOperation() {
  // Run Vector -> Vector transformations ahead of conversion to LLVM.
  {
    OwningRewritePatternList patterns;
    vector::populateVectorToVectorCanonicalizationPatterns(patterns,
                                                           &getContext());
    vector::populateVectorSlicesLoweringPatterns(patterns, &getContext());
    vector::populateVectorContractLoweringPatterns(patterns, &getContext());
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
  {
    OwningRewritePatternList vectorToLoopsPatterns;
    populateVectorToSCFConversionPatterns(
        vectorToLoopsPatterns, &getContext(),
        VectorTransferToSCFOptions().setUnroll(true));
    applyPatternsAndFoldGreedily(getOperation(),
                                 std::move(vectorToLoopsPatterns));
  }

  auto module = getOperation();

  LLVMTypeConverter converter(&getContext());
  converter.addConversion([](Shape::RankedShapeType, SmallVectorImpl<Type> &) {
    return success();
  });

  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateExpandTanhPattern(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateVectorToSCFConversionPatterns(patterns, &getContext());
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateLinalgToLLVMConversionPatterns(converter, patterns);

  // clang-format off
  patterns.insert<
      ConvertBindingOp<IREE::HAL::InterfaceBindingSubspanOp>,
      ConvertBindingOp<IREE::PlaceholderOp>,
    ConvertFunc,
    ConvertTieShapePattern,
    RemoveMakeRankedShape,
    RemoveInterfaceOpPattern,
    ConvertHALInterfaceLoadConstant,
    ConvertWorkgroupInfoOpPattern<
      IREE::HAL::InterfaceWorkgroupIDOp, kIndexWorkGroupId>,
    ConvertWorkgroupInfoOpPattern<
      IREE::HAL::InterfaceWorkgroupCountOp, kIndexWorkGroupCount>,
    ConvertWorkgroupInfoOpPattern<
      IREE::HAL::InterfaceWorkgroupSizeOp, kIndexWorkGroupSize>
  >(&getContext(), converter);
  // clang-format on

  LLVMConversionTarget target(getContext());
  // IREE::HAL::InterfaceOp will be removed after successful conversion of the
  // rest of the IR.
  target.addLegalOp<ModuleOp, ModuleTerminatorOp, IREE::HAL::InterfaceOp,
                    IREE::HAL::InterfaceBindingOp, IREE::HAL::InterfaceEndOp>();
  target.addIllegalDialect<ShapeDialect, StandardOpsDialect, IREEDialect,
                           IREE::HAL::HALDialect, IREE::Flow::FlowDialect>();

  // Don't apply patterns to private function (e.g num_workgroups func).
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
    if (isEntryPoint(funcOp)) return false;
    return true;
  });
  target.addDynamicallyLegalDialect<ShapeDialect, StandardOpsDialect,
                                    IREEDialect, IREE::HAL::HALDialect,
                                    IREE::Flow::FlowDialect>(
      [&](Operation *op) {
        auto funcParent = op->getParentOfType<FuncOp>();
        if (!funcParent) return false;
        if (isEntryPoint(funcParent)) return false;
        return true;
      });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Once we're done with conversion, remove InterfaceOp.
  module.walk([](IREE::HAL::InterfaceOp op) { op.erase(); });
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

static PassRegistration<ConvertToLLVMPass> pass(
    "iree-codegen-convert-to-llvm",
    "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to "
    "LLVMIR dialect",
    [] { return std::make_unique<ConvertToLLVMPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
