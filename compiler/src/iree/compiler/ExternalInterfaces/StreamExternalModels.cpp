// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/StreamExternalModels.h"

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

namespace mlir::iree_compiler {

namespace {

template <typename OpT>
struct AffinityOpAttrExternalModel
    : public IREE::Stream::AffinityOpInterface::ExternalModel<
          AffinityOpAttrExternalModel<OpT>, OpT> {
  static void add(MLIRContext *context) {
    OpT::template attachInterface<AffinityOpAttrExternalModel<OpT>>(*context);
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
    else
      op->removeAttr("stream.affinity");
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
    else
      op->removeAttr("stream.affinity");
  }
};

} // namespace

void registerStreamExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<IREE::Util::UtilDialect>();

  registry.addExtension(
      +[](MLIRContext *context, IREE::Util::UtilDialect *dialect) {
        GlobalOpAffinityAttrExternalModel<IREE::Util::GlobalOp>::add(context);
        AffinityOpAttrExternalModel<IREE::Util::InitializerOp>::add(context);
        AffinityOpAttrExternalModel<IREE::Util::FuncOp>::add(context);
      });
}

} // namespace mlir::iree_compiler
