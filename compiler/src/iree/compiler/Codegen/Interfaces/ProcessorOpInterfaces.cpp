// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

/// Include the generated interface definitions.
#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.cpp.inc"

namespace mlir::iree_compiler {

static unsigned dimToIndex(gpu::Dimension dim) {
  switch (dim) {
  case gpu::Dimension::x:
    return 0;
  case gpu::Dimension::y:
    return 1;
  case gpu::Dimension::z:
    return 2;
  }
  assert(false && "invalid dimension");
  return 0;
}

struct ThreadIdOpInterface
    : public ProcessorIDInterface::ExternalModel<ThreadIdOpInterface,
                                                 gpu::ThreadIdOp> {
  unsigned getDimIndex(Operation *op) const {
    return dimToIndex(cast<gpu::ThreadIdOp>(op).getDimension());
  }
};

struct BlockDimOpInterface
    : public ProcessorCountInterface::ExternalModel<BlockDimOpInterface,
                                                    gpu::BlockDimOp> {
  unsigned getDimIndex(Operation *op) const {
    return dimToIndex(cast<gpu::BlockDimOp>(op).getDimension());
  }
};

struct WorkgroupIdOpInterface
    : public ProcessorIDInterface::ExternalModel<
          WorkgroupIdOpInterface, IREE::HAL::InterfaceWorkgroupIDOp> {
  unsigned getDimIndex(Operation *op) const {
    return cast<IREE::HAL::InterfaceWorkgroupIDOp>(op)
        .getDimension()
        .getZExtValue();
  }
};

struct WorkgroupCountOpInterface
    : public ProcessorCountInterface::ExternalModel<
          WorkgroupCountOpInterface, IREE::HAL::InterfaceWorkgroupCountOp> {
  unsigned getDimIndex(Operation *op) const {
    return cast<IREE::HAL::InterfaceWorkgroupCountOp>(op)
        .getDimension()
        .getZExtValue();
  }
};

struct WorkgroupTileSizeOpInterface
    : public ProcessorTileSizeInterface::ExternalModel<
          WorkgroupTileSizeOpInterface, IREE::HAL::InterfaceWorkgroupSizeOp> {
  unsigned getDimIndex(Operation *op) const {
    return cast<IREE::HAL::InterfaceWorkgroupSizeOp>(op)
        .getDimension()
        .getZExtValue();
  }
};

void registerProcessorOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gpu::GPUDialect *dialect) {
    gpu::ThreadIdOp::attachInterface<ThreadIdOpInterface>(*ctx);
    gpu::BlockDimOp::attachInterface<BlockDimOpInterface>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, IREE::HAL::HALDialect *dialect) {
    IREE::HAL::InterfaceWorkgroupIDOp::attachInterface<WorkgroupIdOpInterface>(
        *ctx);
    IREE::HAL::InterfaceWorkgroupCountOp::attachInterface<
        WorkgroupCountOpInterface>(*ctx);
    IREE::HAL::InterfaceWorkgroupSizeOp::attachInterface<
        WorkgroupTileSizeOpInterface>(*ctx);
  });
}

} // namespace mlir::iree_compiler
