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

//===- CovertToSPIRVPass.cpp - Pass for the final SPIR-V conversion -------===//
//
// This file implements a pass to perform the final conversion to SPIR-V.
// This pass converts remaining interface ops into SPIR-V global variables,
// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
// corresponding SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/CooperativeMatrixAnalysis.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {
//===----------------------------------------------------------------------===//
// Resource and push constant variable utilities
//===----------------------------------------------------------------------===//
// TODO(antiagainst): move these utilities to MLIR core.

/// Returns the pointer type for the push constant storage containing
/// `elementCount` 32-bit integer values.
spirv::PointerType getPushConstantStorageType(unsigned elementCount,
                                              Builder &builder) {
  auto arrayType = spirv::ArrayType::get(
      SPIRVTypeConverter::getIndexType(builder.getContext()), elementCount,
      /*stride=*/4);
  auto structType = spirv::StructType::get({arrayType}, /*LayoutInfo=*/0);
  return spirv::PointerType::get(structType, spirv::StorageClass::PushConstant);
}

/// Returns the push constant varible containing `elementCount` 32-bit integer
/// values in `body`. Returns null op if such an op does not exit.
spirv::GlobalVariableOp getPushConstantVariable(Block &body,
                                                unsigned elementCount) {
  for (auto varOp : body.getOps<spirv::GlobalVariableOp>()) {
    auto ptrType = varOp.type().cast<spirv::PointerType>();
    // Note that Vulkan requires "There must be no more than one push constant
    // block statically used per shader entry point." So we should always reuse
    // the existing one.
    if (ptrType.getStorageClass() == spirv::StorageClass::PushConstant) {
      auto numElements = ptrType.getPointeeType()
                             .cast<spirv::StructType>()
                             .getElementType(0)
                             .cast<spirv::ArrayType>()
                             .getNumElements();
      if (numElements == elementCount) return varOp;
    }
  }
  return nullptr;
}

/// Gets or inserts a global variable for push constant storage containing
/// `elementCount` 32-bit integer values in `block`.
spirv::GlobalVariableOp getOrInsertPushConstantVariable(Location loc,
                                                        Block &block,
                                                        unsigned elementCount) {
  if (auto varOp = getPushConstantVariable(block, elementCount)) return varOp;

  auto builder = OpBuilder::atBlockBegin(&block);
  auto typeAttr =
      TypeAttr::get(getPushConstantStorageType(elementCount, builder));
  StringRef name = "__push_constant_var__";
  return builder.create<spirv::GlobalVariableOp>(loc, typeAttr, name,
                                                 /*initializer=*/nullptr);
}

/// Gets the value at the given `offset` of the push constant storage. A global
/// variable will be created for the push constant storage if not existing. Load
/// ops will be created via the given `builder` to load values from the push
/// constant.
Value getPushConstantValue(Operation *op, unsigned elementCount,
                           unsigned offset, OpBuilder &builder) {
  Location loc = op->getLoc();
  Operation *parent = SymbolTable::getNearestSymbolTable(op->getParentOp());
  if (!parent) {
    op->emitError("expected operation to be within a module-like op");
    return nullptr;
  }

  spirv::GlobalVariableOp varOp = getOrInsertPushConstantVariable(
      loc, parent->getRegion(0).front(), elementCount);

  auto i32Type = SPIRVTypeConverter::getIndexType(builder.getContext());
  Value zeroOp = spirv::ConstantOp::getZero(i32Type, loc, builder);
  Value offsetOp = builder.create<spirv::ConstantOp>(
      loc, i32Type, builder.getI32IntegerAttr(offset));
  auto addrOp = builder.create<spirv::AddressOfOp>(loc, varOp);
  auto acOp = builder.create<spirv::AccessChainOp>(
      loc, addrOp, llvm::makeArrayRef({zeroOp, offsetOp}));
  return builder.create<spirv::LoadOp>(loc, acOp);
}

/// Gets or inserts a resource evariable of the given `type` in `block` and bind
/// it to `set` and `binding`.
spirv::GlobalVariableOp getOrInsertResourceVariable(Location loc, Type type,
                                                    unsigned set,
                                                    unsigned binding,
                                                    Block &block) {
  auto name = llvm::formatv("__resource_var_{0}_{1}__", set, binding).str();
  for (auto varOp : block.getOps<spirv::GlobalVariableOp>()) {
    if (varOp.sym_name() == name) return varOp;
  }

  auto builder = OpBuilder::atBlockBegin(&block);
  return builder.create<spirv::GlobalVariableOp>(loc, type, name, set, binding);
}
}  // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns and pass declarations
//===----------------------------------------------------------------------===//

namespace {
/// Template class for attaching type converter to conversion patterns.
template <typename SourceOp>
class InterfaceOpConversion : public OpConversionPattern<SourceOp> {
 public:
  InterfaceOpConversion(MLIRContext *context, TypeConverter &typeConverter,
                        PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit),
        typeConverter(typeConverter) {}

 protected:
  TypeConverter &typeConverter;
};

/// A pattern to convert hal.interface.load.constant into a sequence of SPIR-V
/// ops to load from a global variable representing the push constant storage.
struct HALInterfaceLoadConstantConverter final
    : public InterfaceOpConversion<IREE::HAL::InterfaceLoadConstantOp> {
  using InterfaceOpConversion<
      IREE::HAL::InterfaceLoadConstantOp>::InterfaceOpConversion;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceLoadConstantOp loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// A pattern to convert iree.placeholdder into a sequence of SPIR-V ops to get
/// the address to a global variable representing the resource buffer.
struct IREEPlaceholderConverter final
    : public InterfaceOpConversion<IREE::PlaceholderOp> {
  using InterfaceOpConversion<IREE::PlaceholderOp>::InterfaceOpConversion;

  LogicalResult matchAndRewrite(
      IREE::PlaceholderOp phOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to lower linalg.reshape to SPIR-V. Since all buffers are linearized
/// in SPIR-V lowering, linalg.reshape becomes a no-op.
// TODO(ravishankarm): Move this into MLIR Core.
struct LinalgReshapeConverter final
    : public SPIRVOpLowering<linalg::ReshapeOp> {
  using SPIRVOpLowering<linalg::ReshapeOp>::SPIRVOpLowering;
  LogicalResult matchAndRewrite(
      linalg::ReshapeOp reshapeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(reshapeOp, operands);
    return success();
  }
};

/// Convert subgroup level vector transfert to SPIR-V cooperative
/// matrix load/store if those are supported.
/// TODO(thomasraoux): Move to MLIR core once this is stable.
template <typename OpTy>
class TransferToCoopMatLoadStore final : public SPIRVOpLowering<OpTy> {
 public:
  TransferToCoopMatLoadStore(
      MLIRContext *context, SPIRVTypeConverter &converter,
      const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis)
      : SPIRVOpLowering<OpTy>(context, converter),
        cooperativeMatrixAnalysis(cooperativeMatrixAnalysis) {}

  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!cooperativeMatrixAnalysis.usesCooperativeMatrixType(op))
      return failure();
    auto loc = op.getLoc();
    auto vecType = op.getVectorType();
    if (vecType.getRank() != 2) return failure();
    // TODO(thomasraoux): use coloumn major operand when TransfertRead +
    // TransposeOp.
    if (!op.permutation_map().isIdentity()) return failure();
    if (op.masked() &&
        llvm::any_of(op.masked()->template cast<ArrayAttr>(),
                     [](mlir::Attribute maskedDim) {
                       return maskedDim.cast<BoolAttr>().getValue();
                     }))
      return failure();
    auto matType = spirv::CooperativeMatrixNVType::get(
        vecType.getElementType(), spirv::Scope::Subgroup, vecType.getDimSize(0),
        vecType.getDimSize(1));
    SmallVector<Value, 4> remappedIndices;
    for (auto i : op.indices())
      remappedIndices.push_back(rewriter.getRemappedValue(i));
    Value ptr = spirv::getElementPtr(
        SPIRVOpLowering<OpTy>::typeConverter, op.getMemRefType(),
        rewriter.getRemappedValue(op.memref()), remappedIndices, loc, rewriter);
    int64_t offset = 0;
    SmallVector<int64_t, 2> strides;
    getStridesAndOffset(op.getMemRefType(), strides, offset);
    auto stride = strides[0];
    if (BaseMemRefType::isDynamicStrideOrOffset(stride)) return failure();
    auto int32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, int32Type, IntegerAttr::get(int32Type, stride));
    auto coloumnMajor = rewriter.create<spirv::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
    replaceTransferOp(op, loc, matType, ptr, strideValue, coloumnMajor,
                      rewriter);
    return success();
  }

 private:
  /// Helper to generate the right load/store instruction and replace the
  /// transfer op.
  void replaceTransferOp(OpTy op, Location loc, Type matType, Value ptr,
                         Value strideValue, Value coloumnMajor,
                         ConversionPatternRewriter &rewriter) const;
  const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis;
};

template <>
void TransferToCoopMatLoadStore<vector::TransferReadOp>::replaceTransferOp(
    vector::TransferReadOp op, Location loc, Type matType, Value ptr,
    Value strideValue, Value coloumnMajor,
    ConversionPatternRewriter &rewriter) const {
  Value load = rewriter.create<spirv::CooperativeMatrixLoadNVOp>(
      loc, matType, ptr, strideValue, coloumnMajor, IntegerAttr());
  rewriter.replaceOp(op, load);
}

template <>
void TransferToCoopMatLoadStore<vector::TransferWriteOp>::replaceTransferOp(
    vector::TransferWriteOp op, Location loc, Type matType, Value ptr,
    Value strideValue, Value coloumnMajor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.create<spirv::CooperativeMatrixStoreNVOp>(
      loc, ptr, rewriter.getRemappedValue(op.vector()), strideValue,
      coloumnMajor, IntegerAttr());
  rewriter.eraseOp(op);
}

/// Convert subgroup level vector contract to SPIR-V cooperative
/// matrix matmuladd.
class VectorContractToCoopMatmul final
    : public SPIRVOpLowering<vector::ContractionOp> {
 public:
  VectorContractToCoopMatmul(
      MLIRContext *context, SPIRVTypeConverter &converter,
      const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis)
      : SPIRVOpLowering<vector::ContractionOp>(context, converter),
        cooperativeMatrixAnalysis(cooperativeMatrixAnalysis) {}

  LogicalResult matchAndRewrite(
      vector::ContractionOp contractOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!cooperativeMatrixAnalysis.usesCooperativeMatrixType(contractOp))
      return failure();
    auto loc = contractOp.getLoc();
    // Check that all the operands are cooperative matrix.
    vector::ContractionOp::Adaptor adaptor(operands);
    auto loadA = adaptor.lhs();
    auto loadB = adaptor.rhs();
    auto loadC = adaptor.acc();
    if (!loadA.getType().isa<spirv::CooperativeMatrixNVType>() ||
        !loadB.getType().isa<spirv::CooperativeMatrixNVType>() ||
        !loadC.getType().isa<spirv::CooperativeMatrixNVType>())
      return failure();
    if (llvm::size(contractOp.masks()) != 0) return failure();
    // Check that this is a matmul operation.
    auto iteratorTypes = contractOp.iterator_types().getValue();
    if (!isParallelIterator(iteratorTypes[0]) ||
        !isParallelIterator(iteratorTypes[1]) ||
        !isReductionIterator(iteratorTypes[2]))
      return failure();
    // Coloumn major matmul should have been lowered to Transpose+contract
    // by this point. Transpose can be handled by load/stoore operations.
    if (!isRowMajorMatmul(contractOp.indexing_maps())) return failure();

    Value matmul = rewriter.create<spirv::CooperativeMatrixMulAddNVOp>(
        loc, loadC.getType(), loadA, loadB, loadC);
    rewriter.replaceOp(contractOp, matmul);
    return success();
  }

 private:
  const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis;
};

/// A pass to perform the SPIR-V conversion.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
struct ConvertToSPIRVPass
    : public PassWrapper<ConvertToSPIRVPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  ConvertToSPIRVPass() {}
  ConvertToSPIRVPass(const ConvertToSPIRVPass &pass) {}
};
}  // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns and pass implementations
//===----------------------------------------------------------------------===//

LogicalResult HALInterfaceLoadConstantConverter::matchAndRewrite(
    IREE::HAL::InterfaceLoadConstantOp loadOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // TODO(#1519): hal.interface.load.constant should point to the
  // hal.interface op.
  auto moduleOp = loadOp.getParentOfType<ModuleOp>();
  auto halInterfaceOps =
      llvm::to_vector<1>(moduleOp.getOps<IREE::HAL::InterfaceOp>());
  assert(halInterfaceOps.size() == 1);

  unsigned elementCount =
      halInterfaceOps.front().push_constants()->getZExtValue();
  unsigned offset = loadOp.offset().getZExtValue();

  // The following function generates SPIR-V ops with i32 types. So it does type
  // "conversion" (index -> i32) implicitly.
  auto value = getPushConstantValue(loadOp, elementCount, offset, rewriter);

  rewriter.replaceOp(loadOp, value);
  return success();
}

LogicalResult IREEPlaceholderConverter::matchAndRewrite(
    IREE::PlaceholderOp phOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto moduleOp = phOp.getParentOfType<ModuleOp>();

  Type convertedType = typeConverter.convertType(phOp.getType());
  if (!convertedType) {
    return phOp.emitError()
           << "SPIRV type conversion failed: " << phOp.getType();
  }
  auto bindingOp = dyn_cast_or_null<IREE::HAL::InterfaceBindingOp>(
      SymbolTable::lookupNearestSymbolFrom(
          phOp, phOp.getAttrOfType<SymbolRefAttr>("binding")));

  spirv::GlobalVariableOp varOp = getOrInsertResourceVariable(
      phOp.getLoc(), convertedType, bindingOp.set().getZExtValue(),
      bindingOp.binding().getZExtValue(), *moduleOp.getBody());

  rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(phOp, varOp);
  return success();
}

static void populateVectorToSPIRVPatterns(
    MLIRContext *context, SPIRVTypeConverter &converter,
    OwningRewritePatternList &patterns,
    const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis) {
  patterns.insert<TransferToCoopMatLoadStore<vector::TransferReadOp>,
                  TransferToCoopMatLoadStore<vector::TransferWriteOp>,
                  VectorContractToCoopMatmul>(context, converter,
                                              cooperativeMatrixAnalysis);
}

void ConvertToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  auto targetAttr = spirv::lookupTargetEnv(moduleOp);
  SPIRVTypeConverter typeConverter(targetAttr);
  ScfToSPIRVContext scfToSPIRVContext;

  OwningRewritePatternList patterns;
  // Pull in GPU patterns to convert processor ID ops and loop ops.
  populateGPUToSPIRVPatterns(context, typeConverter, patterns);
  // Pull in SCF patterns to convert control flow ops.
  populateSCFToSPIRVPatterns(context, typeConverter, scfToSPIRVContext,
                             patterns);
  // Pull in standard patterns to convert arithmetic ops and others.
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);
  // Pull in builtin func to spv.func conversion.
  populateBuiltinFuncToSPIRVPatterns(context, typeConverter, patterns);
  auto &cooperativeMatrixAnalysis = getAnalysis<CooperativeMatrixAnalysis>();
  populateVectorToSPIRVPatterns(context, typeConverter, patterns,
                                cooperativeMatrixAnalysis);
  patterns.insert<HALInterfaceLoadConstantConverter, IREEPlaceholderConverter,
                  LinalgReshapeConverter>(context, typeConverter);

  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetAttr);
  // Disallow all other ops.
  target->markUnknownOpDynamicallyLegal([](Operation *) { return false; });
  SmallVector<FuncOp, 1> functions;
  for (FuncOp fn : moduleOp.getOps<FuncOp>()) {
    if (SymbolTable::getSymbolVisibility(fn) != SymbolTable::Visibility::Public)
      continue;
    functions.push_back(fn);
  }

  for (FuncOp fn : functions)
    if (failed(applyFullConversion(fn, *target, patterns)))
      return signalPassFailure();

  // Collect all SPIR-V ops into a spv.module.
  auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  auto spvModule = builder.create<spirv::ModuleOp>(
      moduleOp.getLoc(), spirv::AddressingModel::Logical,
      spirv::MemoryModel::GLSL450);
  Operation *terminator = spvModule.getBlock().getTerminator();
  Dialect *spvDialect = spvModule.getDialect();
  for (Operation &op : llvm::make_early_inc_range(*moduleOp.getBody())) {
    // Skip the newly created spv.module itself.
    if (&op == spvModule) continue;
    if (op.getDialect() == spvDialect) op.moveBefore(terminator);
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVPass() {
  return std::make_unique<ConvertToSPIRVPass>();
}

static PassRegistration<ConvertToSPIRVPass> pass(
    "iree-codegen-convert-to-spirv",
    "Perform final conversion from builtin/GPU/HAL/standard dialect to SPIR-V "
    "dialect");
}  // namespace iree_compiler
}  // namespace mlir
