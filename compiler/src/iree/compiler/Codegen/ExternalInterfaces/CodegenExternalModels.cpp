// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/CodegenExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// PCF Models
//===----------------------------------------------------------------------===//

struct WorkgroupScopeAttr final
    : IREE::PCF::ScopeAttr::ExternalModel<WorkgroupScopeAttr,
                                          Codegen::WorkgroupAttr> {
  SmallVector<Value> getWorkerCounts(Attribute attr, OpBuilder &builder,
                                     Location loc, int64_t numIds) const {
    auto workgroupAttr = cast<Codegen::WorkgroupAttr>(attr);
    bool linearize = workgroupAttr.getLinearize();

    int64_t numIdsToQuery = linearize ? 3 : std::min<int64_t>(3, numIds);

    SmallVector<Value> counts;
    for (int64_t i = 0, e = numIdsToQuery; i < e; ++i) {
      counts.push_back(
          IREE::HAL::InterfaceWorkgroupCountOp::create(builder, loc, i)
              .getResult());
    }

    // If linearize is true, compute the product of all counts and replace the
    // first ID with that.
    if (linearize && !counts.empty()) {
      Value one = counts.size() > 1
                      ? arith::ConstantIndexOp::create(builder, loc, 1)
                      : Value();
      Value linearizedCount = counts[0];
      for (int64_t i = 1, e = counts.size(); i < e; ++i) {
        linearizedCount =
            arith::MulIOp::create(builder, loc, linearizedCount, counts[i])
                .getResult();
        counts[i] = one;
      }
      counts.front() = linearizedCount;
      counts.resize(numIds > 3 ? 3 : numIds);
    }

    // Pad the outer most sizes with 1.
    if (numIds > 3) {
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      counts.append(numIds - 3, one);
    }

    return counts;
  }
  SmallVector<Value> getWorkerIDs(Attribute attr, OpBuilder &builder,
                                  Location loc, int64_t numIds) const {
    auto workgroupAttr = cast<Codegen::WorkgroupAttr>(attr);
    bool linearize = workgroupAttr.getLinearize();

    SmallVector<Value> ids;
    for (int64_t i = 0, e = std::min<int64_t>(3, numIds); i < e; ++i) {
      ids.push_back(IREE::HAL::InterfaceWorkgroupIDOp::create(builder, loc, i)
                        .getResult());
    }
    if (numIds > 3) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      ids.append(numIds - 3, zero);
    }

    // If linearize is true, flatten to a single ID and pad with `0` until
    // numIds.
    if (linearize && numIds > 1) {
      // First, get the counts to use as the delinearization shape. Construct
      // them in reverse because thats the expected input order for
      // linearize_index.
      SmallVector<Value> dynamicCounts;
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      if (numIds > 3) {
        dynamicCounts.append(numIds - 3, one);
      }
      for (int64_t i = std::min<int64_t>(3, numIds) - 1, e = 0; i >= e; --i) {
        dynamicCounts.push_back(
            IREE::HAL::InterfaceWorkgroupCountOp::create(builder, loc, i)
                .getResult());
      }

      // Linearize. IDs need to be reversed to match the counts and the builder
      // can't take iterators.
      SmallVector<Value> reverseIds(llvm::reverse(ids));
      auto linearizeOp = affine::AffineLinearizeIndexOp::create(
          builder, loc, reverseIds, dynamicCounts);

      ids.front() = linearizeOp.getResult();
      for (int64_t i = 1, e = ids.size(); i < e; ++i) {
        ids[i] = one;
      }
    }

    return ids;
  }
  FailureOr<Attribute> getAllocMemSpace(Attribute, MLIRContext *) const {
    // Allocating workgroup memory unsupported.
    return failure();
  }
};

class CodegenPCFConversionInterface : public PCFConversionDialectInterface {
public:
  using PCFConversionDialectInterface::PCFConversionDialectInterface;
  void
  loadStructuralLoweringDependentDialects(MLIRContext *context) const override {
    // HAL For workgroup ID/Counts.
    context->loadDialect<IREE::HAL::HALDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerCodegenExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        WorkgroupAttr::attachInterface<WorkgroupScopeAttr>(*ctx);
        dialect->addInterface<CodegenPCFConversionInterface>();
      });
}

} // namespace mlir::iree_compiler::IREE::Codegen
