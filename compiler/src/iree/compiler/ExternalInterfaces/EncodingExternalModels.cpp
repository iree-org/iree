// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/EncodingExternalModels.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {
namespace {

struct ContractionAttrPropagationInterface
    : public IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
          ContractionAttrPropagationInterface, IREE::Encoding::MatmulKAttr> {
  bool isPropagable(Attribute attr, Value target) const {
    auto encoding = dyn_cast<IREE::Encoding::MatmulKAttr>(attr);
    if (!encoding) {
      return false;
    }
    if (auto collapseOp = target.getDefiningOp<tensor::CollapseShapeOp>()) {
      ArrayRef<int32_t> kDims = encoding.getKDims().asArrayRef();

      // Bail out if it is not propagable.
      // TODO: Relax the check to allow transforming innermost reduction
      // dimensions. We need to revisit the matmul_k encoding semantic.
      SmallVector<ReassociationIndices, 4> reassociationMaps =
          collapseOp.getReassociationIndices();
      for (int32_t k : kDims) {
        if (reassociationMaps[k].size() != 1) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateEncodings(Attribute attr, Value target) const {
    auto encoding = dyn_cast<IREE::Encoding::MatmulKAttr>(attr);
    if (!encoding) {
      return failure();
    }
    if (auto collapseOp = target.getDefiningOp<tensor::CollapseShapeOp>()) {
      ArrayRef<int32_t> kDims = encoding.getKDims().asArrayRef();
      SmallVector<ReassociationIndices, 4> reassociationMaps =
          collapseOp.getReassociationIndices();
      // Get a mapping from original iteration space to expanded iteration
      // space.
      SmallVector<int32_t> newKDims;
      for (int32_t kDim : kDims) {
        newKDims.append(reassociationMaps[kDim].begin(),
                        reassociationMaps[kDim].end());
      }
      MLIRContext *ctx = collapseOp.getContext();
      auto resultEncodingAttr = IREE::Encoding::MatmulKAttr::get(ctx, newKDims);

      IREE::Encoding::PropagationEncoding propEncoding;
      propEncoding.operandEncodings.push_back(resultEncodingAttr);
      // The result encoding will be the same as the encoding
      // present in the set encoding operation.
      propEncoding.resultEncodings.push_back(encoding);
      return propEncoding;
    }
    return failure();
  }
};

struct ContractionOpPropagationInterface
    : public IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
          ContractionOpPropagationInterface, tensor::CollapseShapeOp> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &builder,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpResult opResult) const {

    auto operandEncodings = encodings.operandEncodings;
    auto resultEncodings = encodings.resultEncodings;

    if (auto collapseShapeOp =
            dyn_cast<tensor::CollapseShapeOp>(opResult.getOwner())) {

      RankedTensorType operandEncodingType =
          collapseShapeOp.getSrcType().cloneWithEncoding(
              operandEncodings.front());

      Value newEncodingOp = builder.create<IREE::Encoding::SetEncodingOp>(
          collapseShapeOp.getLoc(), operandEncodingType,
          collapseShapeOp.getSrc());

      RankedTensorType resultEncodingType =
          dyn_cast<RankedTensorType>(opResult.getType())
              .cloneWithEncoding(resultEncodings.front());

      Value newCollapseOp = builder.create<tensor::CollapseShapeOp>(
          collapseShapeOp.getLoc(), resultEncodingType, newEncodingOp,
          collapseShapeOp.getReassociationIndices());

      IREE::Encoding::PropagationResult result;
      result.replacement = newCollapseOp;
      result.generatedEncodingOps.push_back(newEncodingOp.getDefiningOp());
      return result;
    }
    return failure();
  }
};

} // namespace

void registerEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Encoding::IREEEncodingDialect *dialect) {
        IREE::Encoding::MatmulKAttr::attachInterface<
            ContractionAttrPropagationInterface>(*ctx);
      });
  registry.addExtension(+[](MLIRContext *ctx,
                            mlir::tensor::TensorDialect *dialect) {
    tensor::CollapseShapeOp::attachInterface<ContractionOpPropagationInterface>(
        *ctx);
  });
}

} // namespace mlir::iree_compiler
