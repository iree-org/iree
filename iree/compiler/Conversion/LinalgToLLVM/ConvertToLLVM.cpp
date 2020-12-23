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
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace iree_compiler {

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

// Replace RankedDimOp with resolved index.
class ConvertRankedDimPattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertRankedDimPattern(MLIRContext *context,
                                   LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(Shape::RankedDimOp::getOperationName(), context,
                             typeconverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto rankedDimOp = dyn_cast_or_null<Shape::RankedDimOp>(op);
    if (!rankedDimOp) return failure();
    auto makeRankedShapeOp = dyn_cast_or_null<Shape::MakeRankedShapeOp>(
        rankedDimOp.shape().getDefiningOp());
    if (!makeRankedShapeOp) return failure();
    auto dimIndex = rankedDimOp.index();
    auto dynamicDims =
        makeRankedShapeOp.dynamic_dimensions()[dimIndex.getSExtValue()];
    rewriter.replaceOp(op, dynamicDims);
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

template <typename Op, int ArgIndex>
class ConvertWorkgroupInfoOpPattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertWorkgroupInfoOpPattern(MLIRContext *context,
                                         LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(Op::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = cast<LLVM::LLVMFuncOp>(rewriter.getBlock()->getParentOp());
    auto xyzTy = LLVM::LLVMPointerType::get(
        LLVM::LLVMIntegerType::get(rewriter.getContext(), 32));
    auto xyzArgument = newFuncOp.getArgument(ArgIndex);
    auto dimIndex = rewriter.createOrFold<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMIntegerType::get(rewriter.getContext(), 64),
        op->getAttrOfType<IntegerAttr>("dimension"));
    auto dimPtr = rewriter.createOrFold<LLVM::GEPOp>(
        op->getLoc(), xyzTy, xyzArgument, ValueRange{dimIndex});
    auto dimValue = rewriter.createOrFold<LLVM::LoadOp>(op->getLoc(), dimPtr);
    auto dimValueCasted = rewriter.createOrFold<LLVM::ZExtOp>(
        op->getLoc(), typeConverter->convertType(op->getResult(0).getType()),
        dimValue);
    rewriter.replaceOp(op, dimValueCasted);
    return success();
  }
};

/// Returns true if `aOp` has a desciptor (set, binding) pair smaller than
/// `bOp`. Note that this ignores the offset.
bool operator<(IREE::HAL::InterfaceBindingOp aOp,
               IREE::HAL::InterfaceBindingOp bOp) {
  if (aOp.set() == bOp.set()) return aOp.binding() < bOp.binding();
  return aOp.set() < bOp.set();
}

// Change signature of entry function to func
// clang-format off
// entry_func(%packed_buffers_arg_ptr: !<llvm.int8**>, thread_idx: !llvm.i32, thread_idy: !llvm.i32, thread_idz: !llvm.i32) and lower IREE and HAL ops to
// corresponding LLVMIR ops to construct memref descriptors and load
// push_constant values.
// clang-format on
class ConvertFuncWithHALInterface : public ConvertToLLVMPattern {
 public:
  explicit ConvertFuncWithHALInterface(MLIRContext *context,
                                       LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(mlir::FuncOp::getOperationName(), context,
                             typeconverter, 65535 - 1) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto funcOp = dyn_cast<FuncOp>(op);
    if (!funcOp.isPublic()) return failure();
    FunctionType fnType = funcOp.getType();
    if (fnType.getNumInputs() != 0) {
      return rewriter.notifyMatchFailure(
          funcOp, "entry function should not have inputs");
    }

    // Get interface buffers from all the blocks.
    SmallVector<IREE::PlaceholderOp, 8> bufferOps;
    SmallVector<IREE::HAL::InterfaceLoadConstantOp, 8> loadOps;
    for (Block &block : funcOp.getBlocks()) {
      for (Operation &op : block) {
        if (auto phOp = dyn_cast<IREE::PlaceholderOp>(op)) {
          bufferOps.push_back(phOp);
        } else if (auto phOp =
                       dyn_cast<IREE::HAL::InterfaceLoadConstantOp>(op)) {
          loadOps.push_back(phOp);
        }
      }
    }

    if (bufferOps.empty()) return failure();

    // A map from buffer ops to their corresponding interface binding ops.
    llvm::DenseMap<Operation *, IREE::HAL::InterfaceBindingOp> bufferBindingMap;
    for (auto bufferOp : bufferOps) {
      auto symbol = SymbolTable::lookupNearestSymbolFrom(
          bufferOp, bufferOp->getAttrOfType<SymbolRefAttr>("binding"));
      bufferBindingMap[bufferOp] = cast<IREE::HAL::InterfaceBindingOp>(symbol);
    }

    // Sort buffers according to their descriptor (set, binding) pair.
    llvm::sort(bufferOps, [&bufferBindingMap](IREE::PlaceholderOp aBuffer,
                                              IREE::PlaceholderOp bBuffer) {
      return bufferBindingMap[aBuffer] < bufferBindingMap[bBuffer];
    });

    // A map from buffer ops to their corresponding function argument indices.
    llvm::DenseMap<Operation *, unsigned> bufferArgMap;
    // A map from binding ops to their corresponding function argument indices.
    llvm::DenseMap<Operation *, unsigned> bindingArgMap;
    llvm::SmallVector<MemRefType, 4> inputMemRefTypes;
    llvm::SmallVector<LLVM::LLVMType, 4> inputStructPtrs;
    unsigned argIndex = 0;
    for (auto bufferOp : bufferOps) {
      auto binding = bufferBindingMap[bufferOp];
      auto it = bindingArgMap.find(binding);
      if (it != bindingArgMap.end()) {
        bufferArgMap[bufferOp] = it->second;
      } else {
        bindingArgMap[binding] = argIndex;
        bufferArgMap[bufferOp] = argIndex;
        ++argIndex;
      }

      auto memrefType = bufferOp.getType().dyn_cast_or_null<MemRefType>();
      inputMemRefTypes.push_back(memrefType);
      auto elementType = typeConverter->convertType(memrefType.getElementType())
                             .dyn_cast<LLVM::LLVMType>();
      if (!elementType) return failure();
      inputStructPtrs.push_back(
          LLVM::LLVMPointerType::get(elementType, memrefType.getMemorySpace()));
    }

    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    // clang-format off
    // func foo(%packed_buffer_args: !llvm<i8**>, %push_constant: !llvm<i32*>, thread_idx: i32, thread_idy, thread_idz: i32)
    // clang-format on
    MLIRContext *context = rewriter.getContext();
    auto packedBuffersArgsTy = LLVM::LLVMPointerType::get(
        LLVM::LLVMPointerType::get(LLVM::LLVMIntegerType::get(context, 8)));
    auto pushConstantArgTy =
        LLVM::LLVMPointerType::get(LLVM::LLVMIntegerType::get(context, 32));
    auto xyzTy =
        LLVM::LLVMPointerType::get(LLVM::LLVMIntegerType::get(context, 32));
    signatureConverter.addInputs(packedBuffersArgsTy);
    signatureConverter.addInputs(pushConstantArgTy);
    signatureConverter.addInputs(xyzTy);  // workgroup_id
    signatureConverter.addInputs(xyzTy);  // workgroup_count
    signatureConverter.addInputs(xyzTy);  // workgroup_size

    Location loc = funcOp.getLoc();

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : funcOp.getAttrs()) {
      if (attr.first == SymbolTable::getSymbolAttrName() ||
          attr.first == mlir::impl::getTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    auto newFuncOp = rewriter.create<FuncOp>(
        loc, funcOp.getName(),
        rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                 llvm::None),
        funcAttrs);

    // Move all ops in the old function's region to the new function.
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);

    auto builder = OpBuilder::atBlockBegin(&(newFuncOp.getBlocks().front()),
                                           rewriter.getListener());

    // Cast and unpack input packed_buffer_arguments and construct memref
    // descriptors.
    Value packedBuffersArgsPtr = builder.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
            builder.getContext(), inputStructPtrs)),
        newFuncOp.getArgument(0));
    Value packedBuffersArgs =
        builder.create<LLVM::LoadOp>(loc, packedBuffersArgsPtr);
    for (auto bufferOp : bufferOps) {
      MemRefType memrefType = bufferOp.getType().dyn_cast_or_null<MemRefType>();
      if (!memrefType) return failure();
      const auto index = bufferArgMap[bufferOp];
      Value bufferPtr = builder.create<LLVM::ExtractValueOp>(
          loc, inputStructPtrs[index], packedBuffersArgs,
          rewriter.getI64ArrayAttr(index));
      if (memrefType.hasStaticShape()) {
        auto desc = MemRefDescriptor::fromStaticShape(
            builder, loc, *getTypeConverter(), memrefType, bufferPtr);
        rewriter.replaceOp(bufferOp, {desc});
      } else {
        auto desc = MemRefDescriptor::undef(
            builder, loc, typeConverter->convertType(memrefType));
        desc.setAllocatedPtr(builder, loc, bufferPtr);
        desc.setAlignedPtr(builder, loc, bufferPtr);
        rewriter.replaceOp(bufferOp, {desc});
      }
    }

    // Lower hal.interface.load.constant ops into llvm.getelementptr, llvm.load
    for (auto loadOp : loadOps) {
      Value offset = builder.create<LLVM::ConstantOp>(
          loc, LLVM::LLVMIntegerType::get(context, 64),
          builder.getI64IntegerAttr(loadOp.offset().getZExtValue()));
      Value constPtr = builder.create<LLVM::GEPOp>(loc, pushConstantArgTy,
                                                   newFuncOp.getArgument(1),
                                                   ArrayRef<Value>({offset}));
      Value dimConstant = builder.create<LLVM::LoadOp>(loc, constPtr);
      Value dimConstantCasted = builder.create<LLVM::ZExtOp>(
          loc, typeConverter->convertType(loadOp.getType()), dimConstant);
      rewriter.replaceOp(loadOp, dimConstantCasted);
    }

    rewriter.eraseOp(funcOp);
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

namespace {
struct ConvertToLLVMPass
    : public PassWrapper<ConvertToLLVMPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

}  // namespace

void ConvertToLLVMPass::runOnOperation() {
  // Vector -> Vector transformation is needed before we do any conversion to
  // LLVM.
  {
    OwningRewritePatternList patterns;
    vector::populateVectorToVectorCanonicalizationPatterns(patterns,
                                                           &getContext());
    vector::populateVectorSlicesLoweringPatterns(patterns, &getContext());
    vector::populateVectorContractLoweringPatterns(patterns, &getContext());
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
  //
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
  // The following patterns resolves dynamic shapes by substituting tie_shape
  // ops with an updated memref descriptors and replacing RankDimOp with
  // actual index loaded from memref<?xi32> that holds all dynamic shapes push
  // constants.
  patterns.insert<ConvertFuncWithHALInterface, ConvertRankedDimPattern,
                  ConvertTieShapePattern, RemoveMakeRankedShape,
                  RemoveInterfaceOpPattern>(&getContext(), converter);

  patterns.insert<
      ConvertWorkgroupInfoOpPattern<IREE::HAL::InterfaceWorkgroupIDOp, 2>,
      ConvertWorkgroupInfoOpPattern<IREE::HAL::InterfaceWorkgroupCountOp, 3>,
      ConvertWorkgroupInfoOpPattern<IREE::HAL::InterfaceWorkgroupSizeOp, 4>>(
      &getContext(), converter);

  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // Pass through workspace count calculation. This isn't going to be translated
  // to LLVM.
  // TODO(ataei): Should be handled somewhere else ?
  target.addDynamicallyLegalDialect<ShapeDialect, StandardOpsDialect,
                                    IREEDialect>(
      Optional<ConversionTarget::DynamicLegalityCallbackFn>([](Operation *op) {
        auto funcOp = dyn_cast<FuncOp>(op->getParentOp());
        if (funcOp && !isEntryPoint(funcOp)) return true;
        return false;
      }));

  target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
    bool any = false;
    if (!isEntryPoint(funcOp)) return true;
    funcOp.walk([&](IREE::PlaceholderOp placeholderOp) { any = true; });
    return any ? false : true;
  });
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
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
