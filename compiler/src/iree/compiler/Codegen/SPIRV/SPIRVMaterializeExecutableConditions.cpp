// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {

// Maps the given SPIR-V capability to the corresponding device query used in
// IREE runtime. Returns failure if unsupported yet. Returns nullptr if no need
// for device queries.
//
// Note that the device queries used here should match the ones used in
// iree_hal_vulkan_device_query_i64() on the runtime side.
FailureOr<const char *>
mapToDeviceQuery(spirv::Capability cap,
                 IREE::HAL::ExecutableExportOp entryPoint) {
  switch (cap) {
  case spirv::Capability::Shader:
    // The shader capability is the root capability for graphics APIs.
    // So just ignore.
    return nullptr;

    //===-------------------------------------------------------------------===//
    // Compute capabilities
  case spirv::Capability::Float16:
    return "compute.f16";
  case spirv::Capability::Float64:
    return "compute.f64";
  case spirv::Capability::Int8:
    return "compute.i8";
  case spirv::Capability::Int16:
    return "compute.i16";
  case spirv::Capability::Int64:
    return "compute.i64";

    //===-------------------------------------------------------------------===//
    // Storage capabilities
  case spirv::Capability::UniformAndStorageBuffer8BitAccess:
  case spirv::Capability::StorageBuffer8BitAccess:
    // These capabilities allow 8-bit types to appear in interface variables of
    // a particular storage class.
    // So cluster them together.
    return "storage.8bit";
  case spirv::Capability::StorageBuffer16BitAccess:
  case spirv::Capability::StorageUniform16:
    // These capabilities allow 16-bit types to appear in interface variables of
    // a particular storage class.
    // So cluster them together.
    return "storage.16bit";

    //===-------------------------------------------------------------------===//
    // Subgroup capabilities
  case spirv::Capability::GroupNonUniform:
    // The basic subgroup capability provides access to builtin variables like
    // subgroup ID and size.
    // * In Vulkan, this is mandated starting v1.1.
    // * In Metal, we have it since v2.2.
    // So just ignore.
    return nullptr;
  case spirv::Capability::GroupNonUniformArithmetic:
    return "subgroup.arithmetic";
  case spirv::Capability::GroupNonUniformShuffle:
    return "subgroup.shuffle";

  case spirv::Capability::DotProduct:
  case spirv::Capability::DotProductInput4x8Bit:
    // We only ever use vector<4xi8> -> i32 variant of dot product right now.
    return "dotprod.4xi8.i32";

    //===-------------------------------------------------------------------===//
    // Cooperative matrix capabilities
  case spirv::Capability::CooperativeMatrixKHR: {
    // Cooperative matrix has many device specific configurations. They are not
    // directly reflected in the SPIR-V capabilities. We need to be explicit by
    // looking at the chosen configuration.
    // Format: "coopmatrix.<input-type>.<output-type>.<m>x<n>x<k>".
    auto coopmatType =
        entryPoint->getAttrOfType<ArrayAttr>("iree.spirv.coopmatrix.type");
    auto coopmatShape = entryPoint->getAttrOfType<DenseI64ArrayAttr>(
        "iree.spirv.coopmatrix.shape");
    if (!coopmatType || !coopmatShape)
      return failure();

    Type inputType = cast<TypeAttr>(coopmatType.getValue().front()).getValue();
    Type outputType = cast<TypeAttr>(coopmatType.getValue().back()).getValue();
    int64_t mSize = coopmatShape.asArrayRef()[0];
    int64_t nSize = coopmatShape.asArrayRef()[1];
    int64_t kSize = coopmatShape.asArrayRef()[2];

    // We explicitly perform exact match here given that 1) we need to have the
    // corresponding query in the runtime, and 2) we are not using a lot of
    // configuarations in CodeGen yet.
    if (inputType.isF16() && outputType.isF16()) {
      if (mSize == 16 && nSize == 16 && kSize == 16)
        return "coopmatrix.f16.f16.16x16x16";
    }

    return nullptr;
  }

  default:
    break;
  }
  return failure();
}

struct SPIRVMaterializeExecutableConditionsPass final
    : SPIRVMaterializeExecutableConditionsBase<
          SPIRVMaterializeExecutableConditionsPass> {
  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    if (!usesSPIRVCodeGen(variantOp))
      return;

    IREE::HAL::ExecutableTargetAttr executableTarget = variantOp.getTarget();
    DictionaryAttr configuration = executableTarget.getConfiguration();
    auto spirvTarget = configuration.getAs<spirv::TargetEnvAttr>(
        spirv::getTargetEnvAttrName());

    auto exportOps = variantOp.getOps<IREE::HAL::ExecutableExportOp>();
    if (!llvm::hasSingleElement(exportOps)) {
      variantOp.emitError("expected to contain exactly one export op");
      return signalPassFailure();
    }
    IREE::HAL::ExecutableExportOp exportOp = *exportOps.begin();

    // Map all required SPIR-V capabilities to device queries and unique them.
    // Here we only consider capabilities--version/extension is just the spec
    // "container" for them; so we can ignore.
    SetVector<const char *> queriesSet;
    for (spirv::Capability cap : spirvTarget.getCapabilities()) {
      FailureOr<const char *> query = mapToDeviceQuery(cap, exportOp);
      if (failed(query)) {
        variantOp.emitError("failed to handle capability ")
            << spirv::stringifyCapability(cap);
        return signalPassFailure();
      }
      if (query.value() != nullptr) {
        queriesSet.insert(query.value());
      }
    }

    SmallVector<const char *, 0> queries = queriesSet.takeVector();
    // Sort the vector so we build the hal.executable.condition region in a
    // consistent way to allow comparing later.
    llvm::sort(queries, [](const char *x, const char *y) {
      return StringRef(x) < StringRef(y);
    });

    // Build the hal.executable.condition op inside the variant.
    OpBuilder builder(variantOp);
    Value device = variantOp.createConditionOp(builder);

    IntegerType boolType = builder.getI1Type();
    TypedAttr falseAttr = builder.getBoolAttr(false);
    Location loc = device.getLoc();

    const char *category = "hal.dispatch";

    // Build the condition op region.
    Value result = builder.create<arith::ConstantIntOp>(loc, true, 1);
    for (const char *query : queries) {
      auto queryOp = builder.create<IREE::HAL::DeviceQueryOp>(
          loc, boolType, boolType, device, builder.getStringAttr(category),
          builder.getStringAttr(query), falseAttr);
      // Verify that 1) the query succeeds and 2) the capability is supported.
      auto andOp = builder.create<arith::AndIOp>(loc, queryOp.getOk(),
                                                 queryOp.getValue());
      result = builder.create<arith::AndIOp>(loc, result, andOp);
    }
    builder.create<IREE::HAL::ReturnOp>(loc, result);

    SmallVector<StringRef> features;
    features.reserve(queries.size() + 1);
    features.push_back(variantOp.getTarget().getBackend().getValue());
    for (const char *query : queries) {
      features.push_back(query);
    }

    // Drop the fine-grained SPIR-V target and add the course-grained device
    // queries as a list for the later linking pass to use as a unique key.
    auto dictKeyValues = llvm::to_vector(llvm::make_filter_range(
        configuration.getValue(), [](NamedAttribute attr) {
          return attr.getName() != spirv::getTargetEnvAttrName();
        }));
    dictKeyValues.emplace_back(builder.getStringAttr("iree.spirv.features"),
                               builder.getStrArrayAttr(features));
    variantOp.setTargetAttr(IREE::HAL::ExecutableTargetAttr::get(
        executableTarget.getContext(), executableTarget.getBackend(),
        executableTarget.getFormat(),
        DictionaryAttr::get(configuration.getContext(), dictKeyValues)));
  }
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVMaterializeExecutableConditionsPass() {
  return std::make_unique<SPIRVMaterializeExecutableConditionsPass>();
}

} // namespace mlir::iree_compiler
