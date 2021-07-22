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

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Conversion/TosaToStandard/TosaToStandard.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {
//===----------------------------------------------------------------------===//
// Resource utilities
//===----------------------------------------------------------------------===//

/// Inserts a resource evariable of the given `type` at the beginning of
/// `moduleOp`'s block via `symbolTable` and bind it to `set` and `binding`.
spirv::GlobalVariableOp insertResourceVariable(Location loc, Type type,
                                               unsigned set, unsigned binding,
                                               bool alias, ModuleOp moduleOp,
                                               SymbolTable *symbolTable,
                                               OpBuilder::Listener *listener) {
  OpBuilder builder(moduleOp.getContext(), listener);
  std::string name = llvm::formatv("__resource_var_{0}_{1}_", set, binding);
  auto variable =
      builder.create<spirv::GlobalVariableOp>(loc, type, name, set, binding);
  if (alias) variable->setAttr("aliased", builder.getUnitAttr());
  symbolTable->insert(variable, moduleOp.getBody()->begin());
  return variable;
}

/// Returns the (set, binding) pair for the given interface op.
std::pair<int32_t, int32_t> getInterfaceSetAndBinding(Operation *op) {
  IREE::HAL::InterfaceBindingOp bindingOp =
      cast<IREE::HAL::InterfaceBindingSubspanOp>(op).queryBindingOp();
  return {bindingOp.set().getSExtValue(), bindingOp.binding().getSExtValue()};
}

/// Returns the set of resources that should be marked as aliased in SPIR-V.
llvm::DenseSet<Operation *> getAliasedResources(ModuleOp module) {
  llvm::DenseSet<Operation *> aliasedResources;

  for (FuncOp func : module.getOps<FuncOp>()) {
    // Collect all interface ops and their (set, binding) pairs in this
    // function.
    SmallVector<Operation *, 4> interfaceOps;
    SmallVector<std::pair<uint32_t, uint32_t>, 4> setBindings;
    llvm::DenseMap<std::pair<uint32_t, uint32_t>, unsigned> setBindingCount;
    func.walk([&](Operation *op) {
      if (isa<IREE::HAL::InterfaceBindingSubspanOp>(op)) {
        interfaceOps.emplace_back(op);
        setBindings.emplace_back(getInterfaceSetAndBinding(op));
        ++setBindingCount[setBindings.back()];
      }
    });

    // Perform analysis to determine whether we need to mark the resource as
    // alias. This should happen when we have multiple resources binding to the
    // same (set, binding) pair and they are used in the same function.
    for (unsigned i = 0; i < interfaceOps.size(); ++i) {
      if (setBindingCount[setBindings[i]] > 1) {
        aliasedResources.insert(interfaceOps[i]);
      }
    }
  }

  return aliasedResources;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// A pattern to convert hal.interface.load.constant into a sequence of SPIR-V
/// ops to load from a global variable representing the push constant storage.
struct HALInterfaceLoadConstantConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceLoadConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceLoadConstantOp loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(#1519): hal.interface.load.constant should point to the
    // hal.interface op.
    auto moduleOp = loadOp->getParentOfType<ModuleOp>();
    auto halInterfaceOps =
        llvm::to_vector<1>(moduleOp.getOps<IREE::HAL::InterfaceOp>());
    assert(halInterfaceOps.size() == 1);
    assert(halInterfaceOps.front().push_constants().hasValue());

    uint64_t elementCount =
        (*halInterfaceOps.front().push_constants()).getZExtValue();
    unsigned offset = loadOp.offset().getZExtValue();

    // The following function generates SPIR-V ops with i32 types. So it does
    // type "conversion" (index -> i32) implicitly.
    auto value =
        spirv::getPushConstantValue(loadOp, elementCount, offset, rewriter);

    rewriter.replaceOp(loadOp, value);
    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count into corresponding
/// SPIR-V Builtin ops.
template <typename InterfaceOpTy, spirv::BuiltIn builtin>
struct HALInterfaceWorkgroupIdAndCountConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InterfaceOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    int32_t index = static_cast<int32_t>(op.dimension().getSExtValue());
    Value spirvBuiltin = spirv::getBuiltinVariableValue(op, builtin, rewriter);
    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        op, rewriter.getIntegerType(32), spirvBuiltin,
        rewriter.getI32ArrayAttr({index}));
    return success();
  }
};

/// A pattern to convert hal.interface.binding.subspan into a sequence of SPIR-V
/// ops to get the address to a global variable representing the resource
/// buffer.
struct HALInterfaceBindingSubspanConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  HALInterfaceBindingSubspanConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const llvm::DenseSet<Operation *> &aliasedResources,
      SymbolTable *symbolTable, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        aliasedResources(aliasedResources),
        symbolTable(symbolTable) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp interfaceOp,
      ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = interfaceOp->template getParentOfType<ModuleOp>();

    Type resultType = interfaceOp.getOperation()->getResult(0).getType();
    Type convertedType = this->getTypeConverter()->convertType(resultType);
    if (!convertedType) {
      return interfaceOp.emitError()
             << "failed to convert SPIR-V type: " << resultType;
    }
    auto setAndBinding = getInterfaceSetAndBinding(interfaceOp.getOperation());

    // We always create a new resource variable for the interface.
    spirv::GlobalVariableOp varOp = insertResourceVariable(
        interfaceOp.getLoc(), convertedType, setAndBinding.first,
        setAndBinding.second,
        aliasedResources.contains(interfaceOp.getOperation()), moduleOp,
        symbolTable, rewriter.getListener());

    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(interfaceOp, varOp);
    return success();
  }

 private:
  const llvm::DenseSet<Operation *> &aliasedResources;
  SymbolTable *symbolTable;
};

/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands);
    return success();
  }
};

/// Removes unrealized_conversion_cast ops introduced during progressive
/// lowering when possible.
struct RemoveIdentityConversionCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1 &&
        operands.front().getType() == op->getResultTypes().front()) {
      rewriter.replaceOp(op, operands);
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
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
struct ConvertToSPIRVPass : public ConvertToSPIRVBase<ConvertToSPIRVPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  ConvertToSPIRVPass() {}
  ConvertToSPIRVPass(const ConvertToSPIRVPass &pass) {}

  void runOnOperation() override;
};
}  // namespace

void ConvertToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  auto targetAttr = spirv::lookupTargetEnv(moduleOp);
  SPIRVTypeConverter typeConverter(targetAttr);
  OwningRewritePatternList patterns(&getContext());
  ScfToSPIRVContext scfToSPIRVContext;

  // Pull in GPU patterns to convert processor ID ops and loop ops.
  populateGPUToSPIRVPatterns(typeConverter, patterns);

  // Pull in SCF patterns to convert control flow ops.
  populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);

  // Use the default 64-bit lowering for TOSA's ApplyScale operator:
  //   This lowering widens integer types to 64-bit an performs the non-fused
  //   operations, specifically multiply, add, and shift. Bit-widening
  //   is used to guarantee higher-order bits are not truncated during the
  //   multiply or add.
  //
  // TODO(antiagainst): Use a lowering that uses specific SPIRV intrinsics.
  tosa::populateTosaRescaleToStandardConversionPatterns(&patterns);

  // Pull in standard patterns to convert arithmetic ops and others.
  populateStandardToSPIRVPatterns(typeConverter, patterns);

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

  // Pull in builtin func to spv.func conversion.
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

  // Add IREE HAL interface op conversions.
  patterns.insert<
      HALInterfaceLoadConstantConverter,
      HALInterfaceWorkgroupIdAndCountConverter<
          IREE::HAL::InterfaceWorkgroupIDOp, spirv::BuiltIn::WorkgroupId>,
      HALInterfaceWorkgroupIdAndCountConverter<
          IREE::HAL::InterfaceWorkgroupCountOp, spirv::BuiltIn::NumWorkgroups>>(
      typeConverter, context);

  // Create a symbol table for the current module op. This must be done before
  // starting conversion. The conversion framework has a deferred nature.
  // Actions like replacing/deleting ops are just recorded instead of directly
  // applied. So for function conversion, we will see both the original function
  // and the new replacement function during conversion process. That causes an
  // issue for creating symbol tables, which scans all symbols and errors out
  // when duplicates are found. We should be fine here tough becuase we only use
  // this symbol table to insert new SPIR-V global variables.
  SymbolTable symbolTable(moduleOp);
  auto aliasedResources = getAliasedResources(moduleOp);
  patterns.insert<HALInterfaceBindingSubspanConverter>(
      typeConverter, context, aliasedResources, &symbolTable);

  /// Fold certain operations as no-ops:
  /// - linalg.reshape becomes a no-op since all memrefs are linearized in
  ///   SPIR-V.
  /// - tensor_to_memref can become a no-op since tensors are lowered to
  ///   !spv.array.
  /// - unrealized_conversion_cast with the same source and target type.
  patterns.insert<
      FoldAsNoOp<memref::CollapseShapeOp>, FoldAsNoOp<memref::ExpandShapeOp>,
      FoldAsNoOp<memref::BufferCastOp>, RemoveIdentityConversionCast>(
      typeConverter, context);

  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);
  // Disallow all other ops.
  target->markUnknownOpDynamicallyLegal([](Operation *) { return false; });

  SmallVector<FuncOp, 1> functions;
  for (FuncOp fn : moduleOp.getOps<FuncOp>()) {
    if (!fn.isPublic()) continue;
    functions.push_back(fn);
  }

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (FuncOp fn : functions) {
    if (failed(applyFullConversion(fn, *target, frozenPatterns))) {
      return signalPassFailure();
    }
  }

  // Collect all SPIR-V ops into a spv.module.
  auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  auto spvModule = builder.create<spirv::ModuleOp>(
      moduleOp.getLoc(), spirv::AddressingModel::Logical,
      spirv::MemoryModel::GLSL450);
  Block *body = spvModule.getBody();
  Dialect *spvDialect = spvModule->getDialect();
  for (Operation &op : llvm::make_early_inc_range(*moduleOp.getBody())) {
    // Skip the newly created spv.module itself.
    if (&op == spvModule) continue;
    if (op.getDialect() == spvDialect) op.moveBefore(body, body->end());
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVPass() {
  return std::make_unique<ConvertToSPIRVPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
