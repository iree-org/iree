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

/// Propagate an encoding through an "encoding castable" op. Encoding castable
/// means that the op can be encoded by casting its types to the encoded types.
/// This transform adds iree_encoding.set_encoding ops to the operands op the
/// `op`, and clones the `op` with the new encoded operands and encoded result
/// types. If the `opResult` is produced by an iree_encoding.unset_encoding op,
/// and it is consumed by the `op`, then take the source of the unset encoding
/// instead of re-setting the encoding. If the `opResult` is produced by the
/// `op`, then do not unset the encoding after cloning the op, because the
/// encoded result will be used for propagation.
///
/// Use this function for ops that:
/// 1. Are encoded by casting their types to the encoded types.
/// 2. Are able to directly use the source of any producer unset_encoding ops
///    for propagation, and do not need to re-set the encoding.
static FailureOr<IREE::Encoding::PropagationResult>
propagateThroughEncodingCastableOp(
    RewriterBase &builder, Operation *op,
    IREE::Encoding::PropagationEncoding encodings, OpResult opResult) {
  // We can only set encodings on RankedTensorType.
  if (!llvm::all_of(
          llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()),
          llvm::IsaPred<RankedTensorType>)) {
    return failure();
  }
  SmallVector<Value> encodedOperands;
  IREE::Encoding::PropagationResult result;
  auto maybeUnsetEncodingOp =
      dyn_cast<IREE::Encoding::UnsetEncodingOp>(opResult.getOwner());
  OpBuilder::InsertionGuard guard(builder);
  for (auto [operand, encoding] :
       llvm::zip(op->getOperands(), encodings.operandEncodings)) {
    // If the operand comes from the provided opResult, and the owner of the
    // opResult is an iree_encoding.unset_encoding op, then we don't need to
    // set the encoding again, because this opResult is assumed to be the source
    // of propagation.
    if (operand == opResult && maybeUnsetEncodingOp) {
      encodedOperands.push_back(maybeUnsetEncodingOp.getSource());
      continue;
    }
    auto operandType = cast<RankedTensorType>(operand.getType());
    auto encodedOperandType = operandType.cloneWithEncoding(encoding);
    builder.setInsertionPointAfterValue(operand);
    // Special case for tensor.empty ops, which can simply be cloned with the
    // encoding, instead of creating a new set_encoding op.
    if (auto emptyOp = operand.getDefiningOp<tensor::EmptyOp>()) {
      auto encodedEmptyOp = tensor::EmptyOp::create(
          builder, op->getLoc(), encodedOperandType.getShape(),
          encodedOperandType.getElementType(), emptyOp.getDynamicSizes(),
          encoding);
      encodedOperands.push_back(encodedEmptyOp.getResult());
      continue;
    }
    // Otherwise, we need to create a new set_encoding op.
    auto setEncodingOp = IREE::Encoding::SetEncodingOp::create(
        builder, op->getLoc(), encodedOperandType, operand);
    encodedOperands.push_back(setEncodingOp.getResult());
    result.generatedEncodingOps.push_back(setEncodingOp);
  }
  SmallVector<Type> encodedResultTypes;
  for (auto [result, encoding] :
       llvm::zip(op->getResults(), encodings.resultEncodings)) {
    auto resultType = cast<RankedTensorType>(result.getType());
    auto encodedResultType = resultType.cloneWithEncoding(encoding);
    encodedResultTypes.push_back(encodedResultType);
  }
  builder.setInsertionPoint(op);
  Operation *encodedOp =
      clone(builder, op, encodedResultTypes, encodedOperands);
  for (OpResult encodedResult : encodedOp->getOpResults()) {
    // If this encoded result is coming from the source of propagation, we want
    // to return the encoded result.
    if (op->getOpResult(encodedResult.getResultNumber()) == opResult) {
      result.replacements.push_back(encodedResult);
      continue;
    }
    // Otherwise, we need to unset the encoding so the types are consistent with
    // the other results' users.
    builder.setInsertionPointAfterValue(encodedResult);
    auto unsetEncodingOp = IREE::Encoding::UnsetEncodingOp::create(
        builder, op->getLoc(), encodedResult.getType(), encodedResult);
    result.generatedEncodingOps.push_back(unsetEncodingOp);
    result.replacements.push_back(unsetEncodingOp.getResult());
  }
  return result;
}

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

struct LayoutAttrPropagationInterface final
    : IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
          LayoutAttrPropagationInterface, IREE::Encoding::LayoutAttr> {
  bool isPropagableUp(Attribute attr, OpResult target) const {
    return TypeSwitch<Operation *, bool>(target.getOwner())
        .Case<tensor::CastOp>([&](auto castOp) {
          // CastOp is propagable if it is casting between compatible shapes,
          // because the dimensions need to be consistent with the
          // user_indexing_maps carried by the encoding.
          return verifyCompatibleShape(castOp.getSource().getType(),
                                       castOp.getType())
              .succeeded();
        })
        .Default([&](auto) { return false; });
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateBubblingEncodings(Attribute attr, OpResult target) const {
    auto encoding = cast<IREE::Encoding::LayoutAttr>(attr);
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationEncoding>>(
               target.getOwner())
        .Case<tensor::CastOp>([&](tensor::CastOp) {
          IREE::Encoding::PropagationEncoding propEncoding;
          propEncoding.resultEncodings.push_back(encoding);
          propEncoding.operandEncodings.push_back(encoding);
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
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationResult>>(
               opResult.getOwner())
        .Case<IREE::Encoding::UnsetEncodingOp>(
            [&](auto encodingOp)
                -> FailureOr<IREE::Encoding::PropagationResult> {
              return propagateThroughEncodingCastableOp(rewriter, genericOp,
                                                        encodings, opResult);
              // IREE::Encoding::PropagationResult result;
              // // Set encodings on each input.
              // SmallVector<Value> encodedOperands;
              // encodedOperands.reserve(operandEncodings.size() +
              //                         resultEncodings.size());
              // for (auto [operand, encoding] : llvm::zip(
              //          genericOp.getDpsInputOperands(), operandEncodings)) {
              //   // If the source op is the encoding op, we can just add the
              //   // source to new operands vector and continue.
              //   Operation *sourceOp = operand->get().getDefiningOp();
              //   if (sourceOp && sourceOp == encodingOp) {
              //     encodedOperands.push_back(encodingOp.getSource());
              //     continue;
              //   }

              //   auto operandType =
              //       dyn_cast<RankedTensorType>(operand->get().getType());
              //   if (!operandType) {
              //     // Scalar types do not need encodings.
              //     encodedOperands.push_back(operand->get());
              //     continue;
              //   }
              //   auto resType = RankedTensorType::get(
              //       operandType.getShape(), operandType.getElementType(),
              //       encoding);
              //   Value encodedInput = IREE::Encoding::SetEncodingOp::create(
              //       rewriter, loc, resType, operand->get());
              //   result.generatedEncodingOps.push_back(
              //       encodedInput.getDefiningOp());
              //   encodedOperands.push_back(encodedInput);
              // }

              // SmallVector<Type> resultEncodingTypes;
              // resultEncodingTypes.reserve(resultEncodings.size());
              // for (auto [operand, encoding] :
              //      llvm::zip_equal(genericOp.getDpsInits(), resultEncodings))
              //      {
              //   // Manually cast to work around a gcc bug with type deduction
              //   in
              //   // lambdas.
              //   auto emptyOp =
              //       dyn_cast_or_null<tensor::EmptyOp>(operand.getDefiningOp());
              //   if (!emptyOp) {
              //     return failure();
              //   }
              //   auto resultEncodingType =
              //       dyn_cast<RankedTensorType>(emptyOp.getResult().getType())
              //           .cloneWithEncoding(encoding);

              //   // Create encoded generic op.
              //   rewriter.setInsertionPointAfter(emptyOp);
              //   Value encodedInit = tensor::EmptyOp::create(
              //       rewriter, loc, emptyOp.getType().getShape(),
              //       resultEncodingType.getElementType(),
              //       emptyOp.getDynamicSizes(), encoding);
              //   resultEncodingTypes.push_back(resultEncodingType);
              //   encodedOperands.push_back(encodedInit);
              // }

              // // Create the generic op with new encoded operands.
              // rewriter.setInsertionPointAfter(genericOp);
              // auto encodedGenericOp = clone(
              //     rewriter, genericOp, resultEncodingTypes, encodedOperands);

              // // Create the replacement unset encoding ops.
              // for (OpResult genericResult : encodedGenericOp->getOpResults())
              // {
              //   auto resultType =
              //       cast<RankedTensorType>(genericResult.getType())
              //           .dropEncoding();
              //   auto newUnsetEncoding =
              //   IREE::Encoding::UnsetEncodingOp::create(
              //       rewriter, encodingOp.getLoc(), resultType, genericResult,
              //       encodingOp.getResultDims());
              //   result.replacements.push_back(newUnsetEncoding.getResult());
              //   result.generatedEncodingOps.push_back(newUnsetEncoding);
              // }
              // return result;
            })
        .Default([&](auto) { return failure(); });
  }
};

struct CollapseShapeOpPropagationInterface final
    : IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
          CollapseShapeOpPropagationInterface, tensor::CollapseShapeOp> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &builder,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpResult opResult) const {
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationResult>>(
               opResult.getOwner())
        .Case<tensor::CollapseShapeOp>([&](auto collapseOp) {
          return propagateThroughEncodingCastableOp(builder, collapseOp,
                                                    encodings, opResult);
        })
        .Default([&](auto) { return failure(); });
  }
};

struct CastOpPropagationInterface final
    : IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
          CastOpPropagationInterface, tensor::CastOp> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &builder,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpResult opResult) const {
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationResult>>(
               opResult.getOwner())
        .Case<tensor::CastOp>([&](auto castOp) {
          return propagateThroughEncodingCastableOp(builder, castOp, encodings,
                                                    opResult);
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
        IREE::Encoding::LayoutAttr::attachInterface<
            LayoutAttrPropagationInterface>(*ctx);
      });
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::tensor::TensorDialect *dialect) {
        tensor::CollapseShapeOp::attachInterface<
            CollapseShapeOpPropagationInterface>(*ctx);
        tensor::CastOp::attachInterface<CastOpPropagationInterface>(*ctx);
      });
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::linalg::LinalgDialect *dialect) {
        linalg::GenericOp::attachInterface<GenericOpPropagationInterface>(*ctx);
      });
}

} // namespace mlir::iree_compiler
