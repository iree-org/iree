// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CovertToSPIRVPass.cpp - Performs the final SPIR-V conversion -------===//
//
// This file implements a pass to perform the final conversion to SPIR-V.
// This pass converts remaining interface ops into SPIR-V global variables,
// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
// corresponding SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
#include <tuple>

#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTTOSPIRVPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Resource utilities
//===----------------------------------------------------------------------===//

/// Resource info describing a hal.interface.binding.subspan op.
struct SubspanResourceInfo {
  uint32_t set = 0;
  uint32_t binding = 0;
  Type type = nullptr;
  bool aliased = false;
  spirv::GlobalVariableOp var = nullptr;
};

/// Map from hal.interface.binding.subspan ops to their corresponding
/// spirv.GlobalVariable ops.
using InterfaceResourceMap = llvm::DenseMap<Operation *, SubspanResourceInfo>;

constexpr uint32_t kIndirectBindingsSetIndex = 3;

/// Creates a resource variable using the `globalVariableType` at the beginning
/// of `moduleOp`'s block via `symbolTable` and binds it to `set` and `binding`.
static spirv::GlobalVariableOp
createResourceVariable(Location loc, const SubspanResourceInfo &resource,
                       spirv::PointerType globalVariableType, ModuleOp moduleOp,
                       SymbolTable *symbolTable, bool isIndirect) {
  OpBuilder builder(moduleOp.getContext());
  spirv::GlobalVariableOp variable;
  if (!isIndirect) {
    std::string name =
        llvm::formatv("__resource_var_{}_{}_", resource.set, resource.binding);
    variable = builder.create<spirv::GlobalVariableOp>(
        loc, globalVariableType, name, resource.set, resource.binding);
    if (resource.aliased)
      variable->setAttr("aliased", builder.getUnitAttr());
  } else {
    std::string name =
        llvm::formatv("__resource_var_indirect_{}_", resource.set);
    variable = builder.create<spirv::GlobalVariableOp>(
        loc, globalVariableType, name, kIndirectBindingsSetIndex, resource.set);
  }
  assert(variable);

  symbolTable->insert(variable, moduleOp.getBody()->begin());
  return variable;
}

/// Returns the (set, binding) pair for the given interface op.
static std::pair<uint32_t, uint32_t>
getInterfaceSetAndBinding(IREE::HAL::InterfaceBindingSubspanOp op) {
  return {0, op.getBinding().getSExtValue()};
}

/// Scans all hal.interface.binding.subspan ops in `module`, creates their
/// corresponding spirv.GlobalVariables when needed, and returns the map.
/// The created variables need to have their types fixed later.
/// Assumes direct bindings and creates a global variable for each (set,
/// binding, type) tuple. These globals may alias.
static InterfaceResourceMap createResourceVariables(mlir::ModuleOp module) {
  SymbolTable symbolTable(module);
  InterfaceResourceMap interfaceToResourceInfo;

  // We insert each new global variable at the beginning of the module,
  // therefore, to preserve the original order, we process all functions and all
  // subspan ops in the reverse order.
  auto functions = llvm::to_vector(module.getOps<mlir::FunctionOpInterface>());
  for (auto func : llvm::reverse(functions)) {
    // Collect all interface ops and their (set, binding) pairs in this
    // function. Use SmallVector here for a deterministic order.
    SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps;
    SmallVector<std::pair<uint32_t, uint32_t>> setBindings;

    // Use a map to see if we have different types for one (set, binding) pair,
    // which will require creating multiple SPIR-V global variables.
    llvm::DenseMap<std::pair<uint32_t, uint32_t>, llvm::DenseSet<Type>>
        setBindingTypes;

    func.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      subspanOps.emplace_back(subspanOp);
      setBindings.emplace_back(getInterfaceSetAndBinding(subspanOp));
      setBindingTypes[setBindings.back()].insert(subspanOp.getType());
    });

    // Keep track of created SPIR-V global variables. This allows us to
    // deduplicate when possible to reduce generated SPIR-V blob size.
    llvm::DenseMap<std::tuple<uint32_t, uint32_t, Type>,
                   spirv::GlobalVariableOp>
        resourceVars;

    // We are using a none type for creating the global variable. It's fine.
    // The correctness boundary is the pass. We will fix it up during
    // conversion so it won't leak.
    auto placeholderType = spirv::PointerType::get(
        NoneType::get(module.getContext()), spirv::StorageClass::StorageBuffer);

    for (auto [subspanOp, setBinding] :
         llvm::reverse(llvm::zip_equal(subspanOps, setBindings))) {
      const auto [set, binding] = setBinding;
      Type type = subspanOp.getType();

      // In the *direct* bindings mode, if we have multiple SPIR-V global
      // variables bound to the same (set, binding) pair and they are used in
      // the same function, those variables need to have alias decoration.
      bool aliases = setBindingTypes[{set, binding}].size() > 1;
      std::tuple<uint32_t, uint32_t, Type> key = {set, binding, type};

      // Use placeholder value for the type to create the variable. We will
      // overwrite it with the actual type later.
      spirv::GlobalVariableOp existingVar = resourceVars.lookup(key);
      SubspanResourceInfo resource = {set, binding, type, aliases, existingVar};
      if (!resource.var) {
        resource.var = createResourceVariable(
            subspanOp.getLoc(), resource, placeholderType, module, &symbolTable,
            /*isIndirect=*/false);
        resourceVars[key] = resource.var;
      }
      assert(resource.var);
      interfaceToResourceInfo[subspanOp] = resource;
    }
  }

  return interfaceToResourceInfo;
}

static spirv::PointerType
getGlobalVarTypeForIndirectBinding(MLIRContext *ctx, uint32_t set,
                                   uint32_t maxBinding) {
  auto placeholderResourceType = IntegerType::get(ctx, 32);
  auto memberPtr = spirv::PointerType::get(
      placeholderResourceType, spirv::StorageClass::PhysicalStorageBuffer);
  SmallVector<Type> members(maxBinding + 1, memberPtr);
  // We are creating a top-level Block decorated interface type here, it needs
  // explicit layout per SPIR-V requirements.
  SmallVector<uint32_t> offsets(maxBinding + 1);
  for (int i = 0; i < offsets.size(); ++i) {
    offsets[i] = 8 * i; // Each pointer takes 8 bytes.
  }
  auto structType = spirv::StructType::get(members, offsets);

  return spirv::PointerType::get(structType,
                                 spirv::StorageClass::StorageBuffer);
}

/// Scans all hal.interface.binding.subspan ops in `module`, creates their
/// corresponding spirv.GlobalVariables when needed, and returns the map.
/// Assumes indirect bindings and creates one global variable for each set, with
/// struct members matching the bindings numbers.
static InterfaceResourceMap
createIndirectResourceVariables(mlir::ModuleOp module) {
  SymbolTable symbolTable(module);
  InterfaceResourceMap interfaceToResourceInfo;

  // We insert each new global variable at the begining of the module,
  // therefore, to preserve the original order, we process all functions and all
  // subspan ops in the reverse order.
  auto functions = llvm::to_vector(module.getOps<func::FuncOp>());
  for (func::FuncOp func : llvm::reverse(functions)) {
    // Collect all interface ops and their (set, binding) pairs in this
    // function. Use SmallVector here for a deterministic order.
    SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps;

    // Keep track of the maximum binding index for each set. We need this to
    // know how many members pointers to add to each global variable struct.
    DenseMap<uint32_t, uint32_t> setToMaxBindingIdx;

    func.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      subspanOps.push_back(subspanOp);
      auto [set, binding] = getInterfaceSetAndBinding(subspanOp);
      auto [it, inserted] = setToMaxBindingIdx.try_emplace(set, binding);
      if (!inserted) {
        it->second = std::max(it->second, binding);
      }
    });

    DenseMap<uint32_t, spirv::GlobalVariableOp> setToVar;
    for (IREE::HAL::InterfaceBindingSubspanOp subspanOp :
         llvm::reverse(subspanOps)) {
      auto [set, binding] = getInterfaceSetAndBinding(subspanOp);
      spirv::GlobalVariableOp existingVar = setToVar.lookup(set);
      SubspanResourceInfo resource = {set, binding, subspanOp.getType(), false,
                                      existingVar};
      if (!resource.var) {
        uint32_t maxBindingIdx = setToMaxBindingIdx[set];
        spirv::PointerType globalVarType = getGlobalVarTypeForIndirectBinding(
            module->getContext(), set, maxBindingIdx);

        resource.var =
            createResourceVariable(subspanOp.getLoc(), resource, globalVarType,
                                   module, &symbolTable, /*isIndirect=*/true);
        setToVar[set] = resource.var;
      }
      assert(resource.var);
      interfaceToResourceInfo[subspanOp] = resource;
    }
  }

  return interfaceToResourceInfo;
}

} // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// A pattern to convert hal.interface.constant.load into a sequence of SPIR-V
/// ops to load from a global variable representing the push constant storage.
struct HALInterfaceLoadConstantConverter final
    : OpConversionPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceConstantLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(#1519): this conversion should look up the entry point information
    // to get the total push constant count.
    auto variantOp = loadOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    auto exportOps = llvm::to_vector<1>(variantOp.getExportOps());
    assert(exportOps.size() == 1);
    auto layoutAttr = exportOps.front().getLayout();

    uint64_t elementCount = layoutAttr.getConstants();
    unsigned index = loadOp.getOrdinal().getZExtValue();

    // The following function generates SPIR-V ops with i32 types. So it does
    // type "conversion" (index -> i32) implicitly. This is expected to be
    // paired with a cast (i32 -> index) afterwards.
    IntegerType i32Type = rewriter.getIntegerType(32);
    Value value = spirv::getPushConstantValue(loadOp, elementCount, index,
                                              i32Type, rewriter);

    rewriter.replaceOp(loadOp, value);
    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count/size into
/// corresponding SPIR-V Builtin ops.
template <typename InterfaceOpTy, spirv::BuiltIn builtin>
struct HALInterfaceWorkgroupOpsConverter final
    : OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InterfaceOpTy op, typename InterfaceOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t index = static_cast<int32_t>(op.getDimension().getSExtValue());
    auto i32Type = rewriter.getIntegerType(32);
    Value spirvBuiltin =
        spirv::getBuiltinVariableValue(op, builtin, i32Type, rewriter);
    Value spirvId = rewriter.create<spirv::CompositeExtractOp>(
        spirvBuiltin.getLoc(), i32Type, spirvBuiltin,
        rewriter.getI32ArrayAttr({index}));

    // Casting if Indexing type not 32-bit.
    auto &typeConverter =
        *this->template getTypeConverter<SPIRVTypeConverter>();
    Type indexType = typeConverter.getIndexType();
    if (indexType != i32Type) {
      spirvId = rewriter.create<spirv::UConvertOp>(spirvId.getLoc(), indexType,
                                                   spirvId);
    }
    rewriter.replaceOp(op, spirvId);
    return success();
  }
};

/// A pattern to convert hal.interface.binding.subspan into a sequence of SPIR-V
/// ops to get the address to a global variable representing the resource
/// buffer.
struct HALInterfaceBindingSubspanConverter final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  HALInterfaceBindingSubspanConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const InterfaceResourceMap &interfaceToResourceVars,
      bool useIndirectBindings, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        interfaceToResourceVars(interfaceToResourceVars),
        useIndirectBindings(useIndirectBindings) {}

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (subspanOp.use_empty()) {
      rewriter.eraseOp(subspanOp);
      return success();
    }

    Value offset = subspanOp.getByteOffset();
    APInt offsetInt;
    if (offset && matchPattern(offset, m_ConstantInt(&offsetInt)) &&
        !offsetInt.isZero()) {
      return subspanOp.emitOpError() << "should have no or zero byte offset";
    }

    Type resultType = subspanOp.getType();
    auto convertedType =
        getTypeConverter()->convertType<spirv::PointerType>(resultType);
    if (!convertedType) {
      return subspanOp.emitError()
             << "failed to convert SPIR-V type: " << resultType;
    }
    SubspanResourceInfo info = interfaceToResourceVars.lookup(subspanOp);
    spirv::GlobalVariableOp varOp = info.var;
    assert(varOp);

    if (!useIndirectBindings) {
      // Fix up the variable's type.
      varOp.setTypeAttr(TypeAttr::get(convertedType));

      rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(subspanOp, varOp);
      return success();
    }

    // Handle indirect bindings.
    if (convertedType.getStorageClass() !=
        spirv::StorageClass::PhysicalStorageBuffer) {
      return subspanOp->emitError(
          "indirect bindings require PhysicalStorageBuffer storage class");
    }

    Location loc = subspanOp.getLoc();
    Value globalAddr = rewriter.create<spirv::AddressOfOp>(loc, varOp);
    auto i32Ty = rewriter.getI32Type();
    Value idx = rewriter.create<spirv::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(info.binding));
    auto ptr = rewriter.create<spirv::AccessChainOp>(loc, globalAddr, idx);
    auto addr = rewriter.create<spirv::LoadOp>(loc, ptr);
    assert(cast<spirv::PointerType>(addr.getType()).getStorageClass() ==
               spirv::StorageClass::PhysicalStorageBuffer &&
           "Expected a physical storage buffer pointer");

    // Bitcast the pointer to the correct pointer type. This is allowed for
    // physical storage buffer addresses.
    Value ptrInt = rewriter.create<spirv::ConvertPtrToUOp>(
        loc, rewriter.getI64Type(), addr);
    rewriter.replaceOpWithNewOp<spirv::ConvertUToPtrOp>(subspanOp,
                                                        convertedType, ptrInt);
    return success();
  }

private:
  const InterfaceResourceMap &interfaceToResourceVars;
  const bool useIndirectBindings;
};

/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Removes memref.cast that converts static and dynamic shapes.
struct RemoveStaticDynamicCast final : OpRewritePattern<memref::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto srcType = llvm::cast<MemRefType>(castOp.getSource().getType());
    auto dstType = llvm::cast<MemRefType>(castOp.getType());
    if (srcType.getRank() == 1 && dstType.getRank() == 1 &&
        srcType.hasStaticShape() != dstType.hasStaticShape()) {
      rewriter.replaceOp(castOp, castOp.getSource());
      return success();
    }
    return failure();
  }
};

/// Removes unrealized_conversion_cast ops introduced during progressive
/// lowering when possible.
struct RemoveIdentityConversionCast final
    : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1 &&
        adaptor.getOperands().front().getType() ==
            op->getResultTypes().front()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Conversion pass
//===----------------------------------------------------------------------===//

/// A pass to perform the SPIR-V conversion.
///
/// Converts remaining interface ops into SPIR-V global variables, GPU processor
/// ID ops into SPIR-V global variables, loop/standard ops into corresponding
/// SPIR-V ops.
class ConvertToSPIRVPass final
    : public impl::ConvertToSPIRVPassBase<ConvertToSPIRVPass> {
public:
  using impl::ConvertToSPIRVPassBase<
      ConvertToSPIRVPass>::ConvertToSPIRVPassBase;
  explicit ConvertToSPIRVPass(unsigned indexBits) : indexBits(indexBits) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler)))
      return failure();
    indexBits = indexBitsOption;
    return success();
  }

  void runOnOperation() override;

private:
  // Use 64 bits for index widths.
  unsigned indexBits;
};
} // namespace

void ConvertToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  if (moduleOp.getBody()->empty())
    return;

  bool useIndirectBindings = usesIndirectBindingsAttr(moduleOp);

  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    auto exportOp = getEntryPoint(funcOp);
    if (!exportOp)
      continue;
    if (funcOp->hasAttr(spirv::getEntryPointABIAttrName()))
      continue;
    std::optional<ArrayAttr> workgroupSize = exportOp->getWorkgroupSize();
    if (!workgroupSize) {
      exportOp->emitOpError(
          "expected workgroup_size attribute to be set for SPIR-V lowering");
      return signalPassFailure();
    }
    auto workgroupSize32 =
        llvm::map_to_vector(workgroupSize.value(), [](Attribute v) {
          return static_cast<int32_t>(
              cast<IntegerAttr>(v).getValue().getZExtValue());
        });

    std::optional<APInt> subgroupSize = exportOp->getSubgroupSize();
    std::optional<int> subgroupSize32;
    if (subgroupSize && subgroupSize->isNonNegative()) {
      subgroupSize32 = subgroupSize->getZExtValue();
    }

    funcOp->setAttr(
        spirv::getEntryPointABIAttrName(),
        spirv::getEntryPointABIAttr(context, workgroupSize32, subgroupSize32));
  }

  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    RewritePatternSet shapePatterns(context);
    shapePatterns.add<RemoveStaticDynamicCast>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(shapePatterns)))) {
      funcOp.emitOpError() << "failed running shape patterns";
      return signalPassFailure();
    }
  }

  /// Rewrite extui/si(bitcast) as a mix of vector.shuffle + bitwise arithmetic.
  /// This handles cases like `vector.bitcast i8 to vector<2xi4>` that come from
  /// narrow load emulation by never materializing the sub-byte values. SPIR-V
  /// does not have support for arithmetic on sub-byte types so we currently
  /// rely on this rewrite for the cases seen today.
  /// TODO: Support general emulation of compute on sub-byte types. This is
  /// not mutually exclusive with this pattern, but does mean it is no longer
  /// load bearing.  Also these patterns are already run during
  /// `EmulateNarrotType` pass but dont trigger there due to missing support for
  /// emulation of `vector.transfer_read` in the emulation path. Remove the
  /// patterns from here after that is done.
  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    RewritePatternSet narrowingPatterns(context);
    vector::populateVectorNarrowTypeRewritePatterns(narrowingPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(narrowingPatterns)))) {
      funcOp.emitOpError() << "failed running narrowing patterns";
      return signalPassFailure();
    }
  }

  // Expand any remaining `bf16` `extf` and `trunc` patterns.
  {
    RewritePatternSet patterns(context);
    arith::populateExpandBFloat16Patterns(patterns);
    arith::BitcastOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      moduleOp.emitOpError() << "failed running bf16 extf/trunc patterns";
      return signalPassFailure();
    }
  }

  if (indexBits != 32 && indexBits != 64) {
    moduleOp.emitOpError(
        "only 32-bit or 64-bit indices are supported for SPIR-V");
    return signalPassFailure();
  }
  bool use64bitIndex = indexBits == 64;

  auto targetAttr = moduleOp->getAttrOfType<spirv::TargetEnvAttr>(
      spirv::getTargetEnvAttrName());
  if (!targetAttr) {
    moduleOp.emitOpError("should contain a spirv.target_env attribute");
    return signalPassFailure();
  }
  spirv::TargetEnv targetEnv(targetAttr);

  if (use64bitIndex && !targetEnv.allows(spirv::Capability::Int64)) {
    moduleOp.emitOpError(
        "64-bit indices are not supported for the specified target "
        "environment");
    return signalPassFailure();
  }
  if (useIndirectBindings && !use64bitIndex) {
    moduleOp->emitError("indirect bindings require 64-bit indexing");
    return signalPassFailure();
  }

  if (useIndirectBindings &&
      (!targetEnv.allows({spirv::Capability::PhysicalStorageBufferAddresses,
                          spirv::Capability::Int64}) ||
       !targetEnv.allows(spirv::Extension::SPV_KHR_physical_storage_buffer))) {
    moduleOp.emitOpError("indirect bindings are not supported for the "
                         "specified target environment");
    return signalPassFailure();
  }

  SPIRVConversionOptions options = {};
  options.use64bitIndex = use64bitIndex;

  SPIRVTypeConverter typeConverter(targetAttr, options);

  // Additionally pull in conversion rules for GPU subgroup MMA ops.
  populateMMAToSPIRVCoopMatrixTypeConversion(typeConverter);
  RewritePatternSet patterns(&getContext());
  ScfToSPIRVContext scfToSPIRVContext;

  // Pull in GPU patterns to convert processor ID ops and loop ops.
  populateGPUToSPIRVPatterns(typeConverter, patterns);
  populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns(typeConverter,
                                                        patterns);

  // Pull in SCF patterns to convert control flow ops.
  populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);

  // Pull in MemRef patterns to convert load/store ops.
  populateMemRefToSPIRVPatterns(typeConverter, patterns);

  // Pull in standard/math patterns to convert arithmetic ops and others.
  arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  arith::populateArithToSPIRVPatterns(typeConverter, patterns);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateMathToSPIRVPatterns(typeConverter, patterns);
  populateComplexToSPIRVPatterns(typeConverter, patterns);

  // Pull in standard patterns to convert tensor operations to SPIR-V. These are
  // primarily used to handle tensor-type constants and contain a
  // threshold. Only those constants that are below the threshold are converted
  // to SPIR-V. In IREE we want to control this threshold at Flow level. So set
  // this value arbitrarily high to make sure that everything within a dispatch
  // region is converted.
  mlir::populateTensorToSPIRVPatterns(
      typeConverter, std::numeric_limits<int64_t>::max() / 8, patterns);

  // Pull in vector patterns to convert vector ops.
  mlir::populateVectorToSPIRVPatterns(typeConverter, patterns);

  // Pull in builtin func to spirv.func conversion.
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

  ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);

  // Add IREE HAL interface op conversions.
  patterns.add<
      HALInterfaceLoadConstantConverter,
      HALInterfaceWorkgroupOpsConverter<IREE::HAL::InterfaceWorkgroupIDOp,
                                        spirv::BuiltIn::WorkgroupId>,
      HALInterfaceWorkgroupOpsConverter<IREE::HAL::InterfaceWorkgroupSizeOp,
                                        spirv::BuiltIn::WorkgroupSize>,
      HALInterfaceWorkgroupOpsConverter<IREE::HAL::InterfaceWorkgroupCountOp,
                                        spirv::BuiltIn::NumWorkgroups>>(
      typeConverter, context);

  // Performs a prelimiary step to analyze all hal.interface.binding.subspan ops
  // and creates spirv.GlobalVariables.
  InterfaceResourceMap interfaceToResourceVars =
      useIndirectBindings ? createIndirectResourceVariables(moduleOp)
                          : createResourceVariables(moduleOp);
  // For using use them in conversion.
  patterns.add<HALInterfaceBindingSubspanConverter>(
      typeConverter, context, interfaceToResourceVars, useIndirectBindings);

  /// Fold certain operations as no-ops:
  /// - linalg.reshape becomes a no-op since all memrefs are linearized in
  ///   SPIR-V.
  /// - tensor_to_memref can become a no-op since tensors are lowered to
  ///   !spirv.array.
  /// - unrealized_conversion_cast with the same source and target type.
  patterns.add<
      FoldAsNoOp<memref::CollapseShapeOp>, FoldAsNoOp<memref::ExpandShapeOp>,
      FoldAsNoOp<bufferization::ToMemrefOp>, RemoveIdentityConversionCast>(
      typeConverter, context);

  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);
  // Disallow all other ops.
  target->markUnknownOpDynamicallyLegal([](Operation *) { return false; });

  SmallVector<mlir::FunctionOpInterface, 1> functions;
  for (auto fn : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    if (!fn.isPublic())
      continue;
    functions.push_back(fn);
  }

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (auto fn : functions) {
    if (failed(applyFullConversion(fn, *target, frozenPatterns))) {
      return signalPassFailure();
    }
  }

  auto addressingModel = spirv::AddressingModel::Logical;
  if (useIndirectBindings)
    addressingModel = spirv::AddressingModel::PhysicalStorageBuffer64;

  // Collect all SPIR-V ops into a spirv.module.
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  auto spvModule = builder.create<spirv::ModuleOp>(
      moduleOp.getLoc(), addressingModel, spirv::MemoryModel::GLSL450);
  Block *body = spvModule.getBody();
  Dialect *spvDialect = spvModule->getDialect();
  for (Operation &op : llvm::make_early_inc_range(*moduleOp.getBody())) {
    // Skip the newly created spirv.module itself.
    if (&op == spvModule)
      continue;
    if (op.getDialect() == spvDialect)
      op.moveBefore(body, body->end());
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>>
createConvertToSPIRVPass(unsigned indexBits) {
  return std::make_unique<ConvertToSPIRVPass>(indexBits);
}

} // namespace mlir::iree_compiler
