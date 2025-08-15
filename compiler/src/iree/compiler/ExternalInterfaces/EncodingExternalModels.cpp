// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/EncodingExternalModels.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "iree-encoding-external-models"

namespace mlir::iree_compiler {
namespace {

struct ContractionAttrPropagationInterface final
    : IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
          ContractionAttrPropagationInterface, IREE::Encoding::MatmulKAttr> {
  bool isPropagableUp(Attribute attr, OpResult target) const {
    auto encoding = cast<IREE::Encoding::MatmulKAttr>(attr);
    Operation *attachedToOperation = target.getOwner();
    if (!attachedToOperation) {
      return false;
    }
    return TypeSwitch<Operation *, bool>(attachedToOperation)
        .Case<tensor::CollapseShapeOp>([&](auto collapseOp) {
          ArrayRef<int32_t> kDims = encoding.getKDims().asArrayRef();
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
        })
        .Default([&](auto) { return false; });
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateBubblingEncodings(Attribute attr, OpResult target) const {
    auto encoding = cast<IREE::Encoding::MatmulKAttr>(attr);
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationEncoding>>(
               target.getOwner())
        .Case<tensor::CollapseShapeOp>([&](auto collapseOp) {
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
          auto operandEncodingAttr =
              IREE::Encoding::MatmulKAttr::get(ctx, newKDims);
          IREE::Encoding::PropagationEncoding propEncoding;
          propEncoding.operandEncodings.push_back(operandEncodingAttr);
          // The result encoding will be the same as the encoding
          // present in the set encoding operation.
          propEncoding.resultEncodings.push_back(encoding);
          return propEncoding;
        })
        .Default([&](auto) { return failure(); });
  }
};

struct ContractionOpPropagationInterface final
    : IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
          ContractionOpPropagationInterface, tensor::CollapseShapeOp> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &builder,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpResult opResult) const {
    Location loc = op->getLoc();
    auto operandEncodings = encodings.operandEncodings;
    auto resultEncodings = encodings.resultEncodings;
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationResult>>(
               opResult.getOwner())
        .Case<tensor::CollapseShapeOp>([&](auto collapseOp) {
          RankedTensorType operandEncodingType =
              collapseOp.getSrcType().cloneWithEncoding(
                  operandEncodings.front());
          Value newEncodingOp = builder.create<IREE::Encoding::SetEncodingOp>(
              loc, operandEncodingType, collapseOp.getSrc());
          auto resultEncodingType =
              dyn_cast<RankedTensorType>(opResult.getType())
                  .cloneWithEncoding(resultEncodings.front());
          Value newCollapseOp = builder.create<tensor::CollapseShapeOp>(
              loc, resultEncodingType, newEncodingOp,
              collapseOp.getReassociationIndices());
          IREE::Encoding::PropagationResult result;
          result.replacements = {newCollapseOp};
          result.generatedEncodingOps.push_back(newEncodingOp.getDefiningOp());
          return result;
        })
        .Default([&](auto) { return failure(); });
  }
};

struct EncodingAttrPropagationInterface final
    : IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
          EncodingAttrPropagationInterface, IREE::Encoding::EncodingAttr> {
  bool isPropagableDown(Attribute attr, OpOperand *target) const {
    return TypeSwitch<Operation *, bool>(target->getOwner())
        .Case<linalg::GenericOp>([&](auto genericOp) {
          // Only support parallel generic ops.
          if (genericOp.getNumReductionLoops() != 0) {
            return false;
          }
          // The unset encoding should not be on one of the inits.
          if (genericOp.isDpsInit(target)) {
            return false;
          }
          // Non-projected permutation indexing maps will unlikely get lowered
          // correctly with the encoding.
          if (llvm::any_of(genericOp->getOpOperands(), [&](OpOperand &operand) {
                AffineMap map = genericOp.getMatchingIndexingMap(&operand);
                return !map.isProjectedPermutation();
              })) {
            return false;
          }
          return true;
        })
        .Default([&](auto) { return false; });
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateSinkingEncodings(Attribute attr, OpOperand *target) const {
    auto encoding = cast<IREE::Encoding::EncodingAttr>(attr);
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationEncoding>>(
               target->getOwner())
        .Case<linalg::GenericOp>([&](auto genericOp) {
          IREE::Encoding::PropagationEncoding propEncoding;
          propEncoding.operandEncodings.reserve(genericOp->getNumOperands());
          // Append the target and respective operand's indexing maps to the
          // encoding's indexing maps to create the new encoding.
          AffineMap invTargetIndexingMap = mlir::inversePermutation(
              genericOp.getMatchingIndexingMap(target));
          auto createNewEncoding =
              [&](AffineMap operandMap) -> IREE::Encoding::EncodingAttr {
            IREE::Encoding::EncodingAttr newEncoding = encoding;
            if (!invTargetIndexingMap.isIdentity()) {
              newEncoding = newEncoding.cloneWithNewOperandIndexingMap(
                  invTargetIndexingMap);
            }
            if (!operandMap.isIdentity()) {
              newEncoding =
                  newEncoding.cloneWithNewOperandIndexingMap(operandMap);
            }
            return newEncoding;
          };
          for (OpOperand *operand : genericOp.getDpsInputOperands()) {
            if (operand != target) {
              AffineMap operandMap = genericOp.getMatchingIndexingMap(operand);
              IREE::Encoding::EncodingAttr newEncoding =
                  createNewEncoding(operandMap);
              propEncoding.operandEncodings.push_back(newEncoding);
            } else {
              propEncoding.operandEncodings.push_back(encoding);
            }
          }
          for (OpOperand &operand : genericOp.getDpsInitsMutable()) {
            AffineMap operandMap = genericOp.getMatchingIndexingMap(&operand);
            IREE::Encoding::EncodingAttr newEncoding =
                createNewEncoding(operandMap);
            propEncoding.operandEncodings.push_back(newEncoding);
            propEncoding.resultEncodings.push_back(newEncoding);
          }
          return propEncoding;
        })
        .Default([&](auto) { return failure(); });
  }
};

struct GenericOpPropagationInterface final
    : IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
          GenericOpPropagationInterface, linalg::GenericOp> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &rewriter,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpResult opResult) const {
    OpBuilder::InsertionGuard guard(rewriter);
    auto genericOp = cast<linalg::GenericOp>(op);
    Location loc = op->getLoc();
    auto operandEncodings = encodings.operandEncodings;
    auto resultEncodings = encodings.resultEncodings;
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationResult>>(
               opResult.getOwner())
        .Case<IREE::Encoding::UnsetEncodingOp>(
            [&](auto encodingOp)
                -> FailureOr<IREE::Encoding::PropagationResult> {
              IREE::Encoding::PropagationResult result;
              // Set encodings on each input.
              SmallVector<Value> encodedOperands;
              encodedOperands.reserve(operandEncodings.size() +
                                      resultEncodings.size());
              for (auto [operand, encoding] : llvm::zip(
                       genericOp.getDpsInputOperands(), operandEncodings)) {
                // If the source op is the encoding op, we can just add the
                // source to new operands vector and continue.
                Operation *sourceOp = operand->get().getDefiningOp();
                if (sourceOp && sourceOp == encodingOp) {
                  encodedOperands.push_back(encodingOp.getSource());
                  continue;
                }

                auto operandType =
                    dyn_cast<RankedTensorType>(operand->get().getType());
                if (!operandType) {
                  // Scalar types do not need encodings.
                  encodedOperands.push_back(operand->get());
                  continue;
                }
                auto resType = RankedTensorType::get(
                    operandType.getShape(), operandType.getElementType(),
                    encoding);
                Value encodedInput =
                    rewriter.create<IREE::Encoding::SetEncodingOp>(
                        loc, resType, operand->get());
                result.generatedEncodingOps.push_back(
                    encodedInput.getDefiningOp());
                encodedOperands.push_back(encodedInput);
              }

              SmallVector<Type> resultEncodingTypes;
              resultEncodingTypes.reserve(resultEncodings.size());
              for (auto [operand, encoding] :
                   llvm::zip_equal(genericOp.getDpsInits(), resultEncodings)) {
                // Manually cast to work around a gcc bug with type deduction in
                // lambdas.
                auto emptyOp =
                    dyn_cast_or_null<tensor::EmptyOp>(operand.getDefiningOp());
                if (!emptyOp) {
                  return failure();
                }
                auto resultEncodingType =
                    dyn_cast<RankedTensorType>(emptyOp.getResult().getType())
                        .cloneWithEncoding(encoding);

                // Create encoded generic op.
                rewriter.setInsertionPointAfter(emptyOp);
                Value encodedInit = rewriter.create<tensor::EmptyOp>(
                    loc, emptyOp.getType().getShape(),
                    resultEncodingType.getElementType(),
                    emptyOp.getDynamicSizes(), encoding);
                resultEncodingTypes.push_back(resultEncodingType);
                encodedOperands.push_back(encodedInit);
              }

              // Create the generic op with new encoded operands.
              rewriter.setInsertionPointAfter(genericOp);
              auto encodedGenericOp = clone(
                  rewriter, genericOp, resultEncodingTypes, encodedOperands);

              // Create the replacement unset encoding ops.
              for (OpResult genericResult : encodedGenericOp->getOpResults()) {
                auto resultType =
                    cast<RankedTensorType>(genericResult.getType())
                        .dropEncoding();
                auto newUnsetEncoding =
                    rewriter.create<IREE::Encoding::UnsetEncodingOp>(
                        encodingOp.getLoc(), resultType, genericResult,
                        encodingOp.getResultDims());
                result.replacements.push_back(newUnsetEncoding.getResult());
                result.generatedEncodingOps.push_back(newUnsetEncoding);
              }
              return result;
            })
        .Default([&](auto) { return failure(); });
  }
};

} // namespace

void registerEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Encoding::IREEEncodingDialect *dialect) {
        IREE::Encoding::MatmulKAttr::attachInterface<
            ContractionAttrPropagationInterface>(*ctx);
        IREE::Encoding::EncodingAttr::attachInterface<
            EncodingAttrPropagationInterface>(*ctx);
      });
  registry.addExtension(+[](MLIRContext *ctx,
                            mlir::tensor::TensorDialect *dialect) {
    tensor::CollapseShapeOp::attachInterface<ContractionOpPropagationInterface>(
        *ctx);
  });
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::linalg::LinalgDialect *dialect) {
        linalg::GenericOp::attachInterface<GenericOpPropagationInterface>(*ctx);
      });
}

} // namespace mlir::iree_compiler
