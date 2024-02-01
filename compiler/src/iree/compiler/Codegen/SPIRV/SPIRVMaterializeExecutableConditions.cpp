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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir::iree_compiler {

namespace {

// The list of device features potentially required by a particular kernel.
//
// Note that the fields used here should match the ones used in
// iree_hal_vulkan_device_properties_t on the runtime side.
struct KernelFeatures {
  // Floating-point compute related feature bitfield:
  // * 0b01: f16
  // * 0b10: f64
  // Note that f32 is assumed to always exist and does not appear in this
  // bitfield.
  uint32_t computeFloat;
  // Integer compute related feature bitfield:
  // * 0b001: i8
  // * 0b010: i16
  // * 0b100: i64
  // Note that i32 or i1 is assumed to always exist and does not appear in
  // this bitfield.
  uint32_t computeInt;
  // Storage bitwidth requirement bitfiled:
  // * 0b01: 8-bit
  // * 0b10: 16-bit
  uint32_t storage;
  // Subgroup operation requirement bitfield:
  // * 0b01: subgroup shuffle operations
  // * 0b10: subgroup arithmetic operations
  uint32_t subgroup;
  // Dot product operation requirement bitfield:
  // ("dotprod.<input-type>.<output-type>")
  // * 0b01: dotprod.4xi8.i32
  uint32_t dotProduct;
  // Cooperative matrix requirement bitfield:
  // ("coopmatrix.<input-element-type>.<output-element-type>.<m>x<n>x<k>")
  // * 0b01: coopmatrix.f16.f16.16x16x16
  uint32_t coopMatrix;
  // Physical storage buffer address bitfield
  // ("address.<mode>")
  // * ob01: address.physical64
  uint32_t address;

  KernelFeatures()
      : computeFloat(0), computeInt(0), storage(0), subgroup(0), dotProduct(0),
        coopMatrix(0), address(0) {}

  bool empty() const {
    return computeFloat == 0 && computeInt == 0 && storage == 0 &&
           subgroup == 0 && dotProduct == 0 && coopMatrix == 0 && address == 0;
  }
};

// Maps the given SPIR-V capability to the corresponding device query feature
// and updates features.
//
// Note that the device queries used here should match the ones used in
// iree_hal_vulkan_get_device_properties() on the runtime side.
LogicalResult mapToDeviceQuery(IREE::HAL::ExecutableExportOp entryPoint,
                               spirv::Capability cap,
                               KernelFeatures &features) {
  switch (cap) {
  case spirv::Capability::Shader:
    // The shader capability is the root capability for graphics APIs.
    // So just ignore.
    return success();

    //===-------------------------------------------------------------------===//
    // Compute capabilities
  case spirv::Capability::Float16:
    features.computeFloat |= 0b01;
    return success();
  case spirv::Capability::Float64:
    features.computeFloat |= 0b10;
    return success();
  case spirv::Capability::Int8:
    features.computeInt |= 0b001;
    return success();
  case spirv::Capability::Int16:
    features.computeInt |= 0b010;
    return success();
  case spirv::Capability::Int64:
    features.computeInt |= 0b100;
    return success();

    //===-------------------------------------------------------------------===//
    // Storage capabilities
  case spirv::Capability::UniformAndStorageBuffer8BitAccess:
  case spirv::Capability::StorageBuffer8BitAccess:
    // These capabilities allow 8-bit types to appear in interface variables of
    // a particular storage class.
    // So cluster them together.
    features.storage |= 0b01;
    return success();
  case spirv::Capability::StorageBuffer16BitAccess:
  case spirv::Capability::StorageUniform16:
    // These capabilities allow 16-bit types to appear in interface variables of
    // a particular storage class.
    // So cluster them together.
    features.storage |= 0b10;
    return success();

    //===-------------------------------------------------------------------===//
    // Subgroup capabilities
  case spirv::Capability::GroupNonUniform:
    // The basic subgroup capability provides access to builtin variables like
    // subgroup ID and size.
    // * In Vulkan, this is mandated starting v1.1.
    // * In Metal, we have it since v2.2.
    // So just ignore.
    return success();
  case spirv::Capability::GroupNonUniformShuffle:
    features.subgroup |= 0b01;
    return success();
  case spirv::Capability::GroupNonUniformArithmetic:
    features.subgroup |= 0b10;
    return success();

  case spirv::Capability::DotProduct:
  case spirv::Capability::DotProductInput4x8Bit:
    // We only ever use vector<4xi8> -> i32 variant of dot product right now.
    features.dotProduct |= 0b1;
    return success();

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
      if (mSize == 16 && nSize == 16 && kSize == 16) {
        features.coopMatrix |= 0b1;
        return success();
      }
    }

    return success();
  }

    //===-------------------------------------------------------------------===//
    // Address capabilities
  case spirv::Capability::PhysicalStorageBufferAddresses:
    // Vulkan only supports 64-bit device buffer addresses.
    features.address |= 0b01;
    return success();

  default:
    break;
  }
  return failure();
}

// Builds the device query ops using the given builder.
//
// Note that the device queries used here should match the ones used in
// iree_hal_vulkan_device_query_i64() on the runtime side.
void buildDeviceQueryRegion(const KernelFeatures &features, Value device,
                            Location loc, OpBuilder &builder) {
  IntegerType boolType = builder.getI1Type();
  IntegerType i32Type = builder.getI32Type();
  TypedAttr zeroAttr = builder.getZeroAttr(i32Type);

  auto buildQueryOp = [&](const char *key, uint32_t value, Value result) {
    auto queryOp = builder.create<IREE::HAL::DeviceQueryOp>(
        loc, boolType, i32Type, device, builder.getStringAttr("hal.dispatch"),
        builder.getStringAttr(key), zeroAttr);
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto val = builder.create<arith::ConstantIntOp>(loc, value, 32);
    auto andOp = builder.create<arith::AndIOp>(loc, queryOp.getValue(), val);
    auto cmpOp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               andOp, zero);
    // Verify that 1) the query succeeds and 2) the capability is supported.
    auto ok = builder.create<arith::AndIOp>(loc, queryOp.getOk(), cmpOp);
    return builder.create<arith::AndIOp>(loc, result, ok).getResult();
  };

  Value result = builder.create<arith::ConstantIntOp>(loc, true, 1);
  if (features.computeFloat) {
    result =
        buildQueryOp("compute.bitwidths.fp", features.computeFloat, result);
  }
  if (features.computeInt) {
    result = buildQueryOp("compute.bitwidths.int", features.computeInt, result);
  }
  if (features.storage) {
    result = buildQueryOp("storage.bitwidths", features.storage, result);
  }
  if (features.subgroup) {
    result = buildQueryOp("subgroup.ops", features.subgroup, result);
  }
  if (features.dotProduct) {
    result = buildQueryOp("dotprod.ops", features.dotProduct, result);
  }
  if (features.coopMatrix) {
    result = buildQueryOp("coopmatrix.ops", features.coopMatrix, result);
  }
  if (features.address) {
    result = buildQueryOp("address.mode", features.address, result);
  }
  builder.create<IREE::HAL::ReturnOp>(loc, result);
}

// Returns the device queries as a list of unique keys.
SmallVector<std::string> getDeviceQueries(const KernelFeatures &features) {
  SmallVector<std::string> queries;
  if (features.computeFloat) {
    queries.push_back("compute.bitwidths.fp=" +
                      std::to_string(features.computeFloat));
  }
  if (features.computeInt) {
    queries.push_back("compute.bitwidths.int=" +
                      std::to_string(features.computeInt));
  }
  if (features.storage) {
    queries.push_back("storage.bitwidths=" + std::to_string(features.storage));
  }
  if (features.subgroup) {
    queries.push_back("subgroup.ops=" + std::to_string(features.subgroup));
  }
  if (features.dotProduct) {
    queries.push_back("dotprod.ops=" + std::to_string(features.dotProduct));
  }
  if (features.coopMatrix) {
    queries.push_back("coopmatrix.ops=" + std::to_string(features.coopMatrix));
  }
  if (features.address) {
    queries.push_back("address.mode=" + std::to_string(features.address));
  }
  return queries;
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
    KernelFeatures features;
    for (spirv::Capability cap : spirvTarget.getCapabilities()) {
      if (failed(mapToDeviceQuery(exportOp, cap, features))) {
        variantOp.emitError("failed to handle capability ")
            << spirv::stringifyCapability(cap);
        return signalPassFailure();
      }
    }

    OpBuilder builder(variantOp);

    // Build the hal.executable.condition op inside the variant.
    if (!features.empty()) {
      Value device = variantOp.createConditionOp(builder);
      buildDeviceQueryRegion(features, device, device.getLoc(), builder);
    }

    // Build a string list of the used queries too--this is useful for attaching
    // to the executable target attribute as a unique key for the linking pass.
    SmallVector<std::string> strings = getDeviceQueries(features);
    SmallVector<StringRef> queries;
    queries.reserve(strings.size() + 1);
    queries.push_back(variantOp.getTarget().getBackend().getValue());
    for (const std::string &s : strings) {
      queries.push_back(s);
    }

    // Drop the fine-grained SPIR-V target and add the course-grained device
    // queries as a list.
    auto dictKeyValues = llvm::to_vector(llvm::make_filter_range(
        configuration.getValue(), [](NamedAttribute attr) {
          return attr.getName() != spirv::getTargetEnvAttrName();
        }));
    dictKeyValues.emplace_back(builder.getStringAttr("iree.spirv.features"),
                               builder.getStrArrayAttr(queries));
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
