// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/StreamExternalModels.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

namespace mlir::iree_compiler {

namespace {

template <typename OpT>
struct PreferCloneToConsumersStreamableOpExternalModel
    : public IREE::Stream::StreamableOpInterface::ExternalModel<
          PreferCloneToConsumersStreamableOpExternalModel<OpT>, OpT> {
  static void add(MLIRContext *context) {
    OpT::template attachInterface<
        PreferCloneToConsumersStreamableOpExternalModel<OpT>>(*context);
  }

  bool preferCloneToConsumers(Operation *op) const { return true; }
};

struct FlowDispatchStreamableOpExternalModel
    : public IREE::Stream::StreamableOpInterface::ExternalModel<
          FlowDispatchStreamableOpExternalModel, IREE::Flow::DispatchOp> {
  static void add(MLIRContext *context) {
    IREE::Flow::DispatchOp::attachInterface<
        FlowDispatchStreamableOpExternalModel>(*context);
  }

  bool preferCloneToConsumers(Operation *op) const {
    // If the dispatch does not consume any resources then it is effectively a
    // slow splat and should be treated like one.
    const bool consumesAny = llvm::any_of(
        op->getOperandTypes(), +[](Type type) {
          return isa<IREE::Stream::AffinityTypeInterface>(type);
        });
    return !consumesAny;
  }
};

template <typename OpT>
struct OptionalOpAffinityAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          OptionalOpAffinityAttrExternalModel<OpT>, OpT> {
  static void add(MLIRContext *context) {
    OpT::template attachInterface<OptionalOpAffinityAttrExternalModel<OpT>>(
        *context);
  }

  // Affinity only required for results that hold resources that
  // require placement.
  bool requiresAffinity(Operation *op) const {
    auto resultType = cast<OpT>(op).getResult().getType();
    return isa<TensorType>(resultType);
  }

  IREE::Stream::AffinityAttr getAffinityAttr(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("stream.affinity");
  }

  void setAffinityAttr(Operation *op, IREE::Stream::AffinityAttr value) const {
    if (value) {
      op->setAttr("stream.affinity", value);
    } else {
      op->removeAttr("stream.affinity");
    }
  }
};

struct FlowBarrierTargetAffinityAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          FlowBarrierTargetAffinityAttrExternalModel,
          IREE::Flow::TensorBarrierOp> {
  static void add(MLIRContext *context) {
    IREE::Flow::TensorBarrierOp::attachInterface<
        FlowBarrierTargetAffinityAttrExternalModel>(*context);
  }

  bool requiresAffinity(Operation *op) const { return true; }

  bool pinsValueAffinity(Operation *op) const {
    return op->hasAttrOfType<IREE::Stream::AffinityAttr>("target");
  }

  IREE::Stream::AffinityAttr getAffinityAttr(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("target");
  }

  void setAffinityAttr(Operation *op, IREE::Stream::AffinityAttr value) const {
    op->setAttr("target", value);
  }
};

struct FlowTransferTargetAffinityAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          FlowTransferTargetAffinityAttrExternalModel,
          IREE::Flow::TensorTransferOp> {
  static void add(MLIRContext *context) {
    IREE::Flow::TensorTransferOp::attachInterface<
        FlowTransferTargetAffinityAttrExternalModel>(*context);
  }

  bool requiresAffinity(Operation *op) const { return true; }

  IREE::Stream::AffinityAttr getAffinityAttr(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("target");
  }

  void setAffinityAttr(Operation *op, IREE::Stream::AffinityAttr value) const {
    op->setAttr("target", value);
  }
};

template <typename OpT>
struct HALTensorAffinityAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          HALTensorAffinityAttrExternalModel<OpT>, OpT> {
  static void add(MLIRContext *context) {
    OpT::template attachInterface<HALTensorAffinityAttrExternalModel<OpT>>(
        *context);
  }

  bool requiresAffinity(Operation *op) const { return false; }

  bool pinsValueAffinity(Operation *op) const {
    return op->hasAttrOfType<IREE::Stream::AffinityAttr>("affinity");
  }

  IREE::Stream::AffinityAttr getAffinityAttr(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("affinity");
  }

  void setAffinityAttr(Operation *op, IREE::Stream::AffinityAttr value) const {
    if (value) {
      op->setAttr("affinity", value);
    } else {
      op->removeAttr("affinity");
    }
  }
};

template <typename OpT>
struct GlobalOpAffinityAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          GlobalOpAffinityAttrExternalModel<OpT>, OpT> {
  static void add(MLIRContext *context) {
    OpT::template attachInterface<GlobalOpAffinityAttrExternalModel<OpT>>(
        *context);
  }

  // Affinity only required for globals that hold resources that require
  // placement.
  bool requiresAffinity(Operation *op) const {
    auto globalType = cast<IREE::Util::GlobalOpInterface>(op).getGlobalType();
    return isa<TensorType>(globalType);
  }

  IREE::Stream::AffinityAttr getAffinityAttr(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("stream.affinity");
  }

  void setAffinityAttr(Operation *op, IREE::Stream::AffinityAttr value) const {
    if (value) {
      op->setAttr("stream.affinity", value);
    } else {
      op->removeAttr("stream.affinity");
    }
  }
};

template <typename OpT, bool kRequiresAffinity = true>
struct AffinityOpAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          AffinityOpAttrExternalModel<OpT, kRequiresAffinity>, OpT> {
  static void add(MLIRContext *context) {
    OpT::template attachInterface<
        AffinityOpAttrExternalModel<OpT, kRequiresAffinity>>(*context);
  }

  // Most structural ops don't require affinities and after placement we don't
  // use the affinities even if the ops still exist.
  bool requiresAffinity(Operation *op) const { return kRequiresAffinity; }

  IREE::Stream::AffinityAttr getAffinityAttr(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("stream.affinity");
  }

  void setAffinityAttr(Operation *op, IREE::Stream::AffinityAttr value) const {
    if (value) {
      op->setAttr("stream.affinity", value);
    } else {
      op->removeAttr("stream.affinity");
    }
  }
};

struct TensorAffinityTypeExternalModel
    : public IREE::Stream::AffinityTypeInterface::ExternalModel<
          TensorAffinityTypeExternalModel, RankedTensorType> {
  static void add(MLIRContext *context) {
    RankedTensorType::attachInterface<TensorAffinityTypeExternalModel>(
        *context);
  }
};

} // namespace

void registerStreamExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context) {
    TensorAffinityTypeExternalModel::add(context);
  });

  registry.insert<arith::ArithDialect>();
  registry.addExtension(
      +[](MLIRContext *context, arith::ArithDialect *dialect) {
        OptionalOpAffinityAttrExternalModel<arith::ConstantOp>::add(context);
      });

  registry.insert<IREE::Flow::FlowDialect>();
  registry.addExtension(+[](MLIRContext *context,
                            IREE::Flow::FlowDialect *dialect) {
    PreferCloneToConsumersStreamableOpExternalModel<
        IREE::Flow::TensorReshapeOp>::add(context);
    PreferCloneToConsumersStreamableOpExternalModel<
        IREE::Flow::TensorAllocaOp>::add(context);
    PreferCloneToConsumersStreamableOpExternalModel<
        IREE::Flow::TensorEmptyOp>::add(context);
    PreferCloneToConsumersStreamableOpExternalModel<
        IREE::Flow::TensorSplatOp>::add(context);
    FlowDispatchStreamableOpExternalModel::add(context);

    FlowBarrierTargetAffinityAttrExternalModel::add(context);
    FlowTransferTargetAffinityAttrExternalModel::add(context);

    AffinityOpAttrExternalModel<IREE::Flow::DispatchRegionOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::DispatchWorkgroupsOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::DispatchOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::CallOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorConstantOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorDynamicConstantOp>::add(
        context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorAllocaOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorEmptyOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorSplatOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorCloneOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorEncodeOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorSliceOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::TensorUpdateOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::ChannelDefaultOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::CollectiveAllGatherOp>::add(
        context);
    AffinityOpAttrExternalModel<IREE::Flow::CollectiveAllReduceOp>::add(
        context);
    AffinityOpAttrExternalModel<IREE::Flow::CollectiveAllToAllOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Flow::CollectiveReduceScatterOp>::add(
        context);
    AffinityOpAttrExternalModel<IREE::Flow::CollectiveSendRecvOp>::add(context);
  });

  registry.insert<IREE::HAL::HALDialect>();
  registry.addExtension(+[](MLIRContext *context,
                            IREE::HAL::HALDialect *dialect) {
    HALTensorAffinityAttrExternalModel<IREE::HAL::TensorImportOp>::add(context);
    HALTensorAffinityAttrExternalModel<IREE::HAL::TensorExportOp>::add(context);
    HALTensorAffinityAttrExternalModel<IREE::HAL::TensorAliasOp>::add(context);
  });

  registry.insert<IREE::Util::UtilDialect>();
  registry.addExtension(+[](MLIRContext *context,
                            IREE::Util::UtilDialect *dialect) {
    GlobalOpAffinityAttrExternalModel<IREE::Util::GlobalOp>::add(context);
    AffinityOpAttrExternalModel<IREE::Util::InitializerOp, false>::add(context);
    AffinityOpAttrExternalModel<IREE::Util::FuncOp, false>::add(context);
  });
}

} // namespace mlir::iree_compiler
