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

struct FlowTransferTargetAffinityAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          FlowTransferTargetAffinityAttrExternalModel,
          IREE::Flow::TensorTransferOp> {
  static void add(MLIRContext *context) {
    IREE::Flow::TensorTransferOp::attachInterface<
        FlowTransferTargetAffinityAttrExternalModel>(*context);
  }

  bool requiresAffinity(Operation *op) const { return true; }

  IREE::Stream::AffinityAttr getAffinity(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("target");
  }

  void setAffinity(Operation *op, IREE::Stream::AffinityAttr value) const {
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

  IREE::Stream::AffinityAttr getAffinity(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("affinity");
  }

  void setAffinity(Operation *op, IREE::Stream::AffinityAttr value) const {
    if (value)
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

  IREE::Stream::AffinityAttr getAffinity(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("stream.affinity");
  }

  void setAffinity(Operation *op, IREE::Stream::AffinityAttr value) const {
    if (value)
      op->setAttr("stream.affinity", value);
    } else {
      op->removeAttr("stream.affinity");
    }
  }
};

template <typename OpT>
struct AffinityOpAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          AffinityOpAttrExternalModel<OpT, kRequiresAffinity>, OpT> {
  static void add(MLIRContext *context) {
    OpT::template attachInterface<
        AffinityOpAttrExternalModel<OpT, kRequiresAffinity>>(*context);
  }

  // Most structural ops don't require affinities and after placement we don't
  // use the affinities even if the ops still exist.
  bool requiresAffinity(Operation *op) const { return false; }

  IREE::Stream::AffinityAttr getAffinity(Operation *op) const {
    return op->getAttrOfType<IREE::Stream::AffinityAttr>("stream.affinity");
  }

  void setAffinity(Operation *op, IREE::Stream::AffinityAttr value) const {
    if (value)
      op->setAttr("stream.affinity", value);
    } else {
      op->removeAttr("stream.affinity");
    }
  }
};

} // namespace

void registerStreamExternalModels(DialectRegistry &registry) {
  registry.insert<IREE::Flow::FlowDialect>();
  registry.addExtension(
      +[](MLIRContext *context, IREE::Flow::FlowDialect *dialect) {
        FlowTransferTargetAffinityAttrExternalModel::add(context);
      });

  registry.insert<IREE::HAL::HALDialect>();
  registry.addExtension(+[](MLIRContext *context,
                            IREE::HAL::HALDialect *dialect) {
    HALTensorAffinityAttrExternalModel<IREE::HAL::TensorImportOp>::add(context);
    HALTensorAffinityAttrExternalModel<IREE::HAL::TensorExportOp>::add(context);
    HALTensorAffinityAttrExternalModel<IREE::HAL::TensorAliasOp>::add(context);
    HALTensorAffinityAttrExternalModel<IREE::HAL::TensorBarrierOp>::add(
        context);
  });

  registry.insert<IREE::Util::UtilDialect>();
  registry.addExtension(
      +[](MLIRContext *context, IREE::Util::UtilDialect *dialect) {
        GlobalOpAffinityAttrExternalModel<IREE::Util::GlobalOp>::add(context);
        AffinityOpAttrExternalModel<IREE::Util::InitializerOp>::add(context);
        AffinityOpAttrExternalModel<IREE::Util::FuncOp>::add(context);
      });
}

} // namespace mlir::iree_compiler
