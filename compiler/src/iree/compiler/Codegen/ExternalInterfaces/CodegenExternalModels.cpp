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

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// PCF Models
//===----------------------------------------------------------------------===//

struct WorkgroupScopeAttr final
    : IREE::PCF::ScopeAttr::ExternalModel<WorkgroupScopeAttr,
                                          Codegen::WorkgroupAttr> {
  SmallVector<Value> getWorkerCounts(Attribute attr, OpBuilder &builder,
                                     Location loc, int64_t numIds) const {
    SmallVector<Value> counts;
    if (numIds > 3) {
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      counts.append(numIds - 3, one);
    }
    for (int64_t i = std::min<int64_t>(3, numIds) - 1; i >= 0; --i) {
      counts.push_back(
          IREE::HAL::InterfaceWorkgroupCountOp::create(builder, loc, i)
              .getResult());
    }
    return counts;
  }
  SmallVector<Value> getWorkerIDs(Attribute attr, OpBuilder &builder,
                                  Location loc, int64_t numIds) const {
    SmallVector<Value> ids;
    if (numIds > 3) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      ids.append(numIds - 3, zero);
    }
    for (int64_t i = std::min<int64_t>(3, numIds) - 1; i >= 0; --i) {
      ids.push_back(IREE::HAL::InterfaceWorkgroupIDOp::create(builder, loc, i)
                        .getResult());
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
