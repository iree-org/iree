// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"

/// Include the generated interface definitions.
#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.cpp.inc"

namespace mlir {
namespace iree_compiler {

static unsigned dimToIndex(gpu::Dimension dim) {
  switch (dim) {
    case gpu::Dimension::x:
      return 0;
    case gpu::Dimension::y:
      return 1;
    case gpu::Dimension::z:
      return 2;
    default:
      llvm_unreachable("invalid dimension");
      return 0;
  }
}

struct ThreadIdOpInterface
    : public ProcessorIDInterface::ExternalModel<ThreadIdOpInterface,
                                                 gpu::ThreadIdOp> {
  unsigned getDimIndex(Operation *op) const {
    return dimToIndex(cast<gpu::ThreadIdOp>(op).dimension());
  }
};

struct BlockDimOpInterface
    : public ProcessorCountInterface::ExternalModel<BlockDimOpInterface,
                                                    gpu::BlockDimOp> {
  unsigned getDimIndex(Operation *op) const {
    return dimToIndex(cast<gpu::BlockDimOp>(op).dimension());
  }
};

struct WorkgroupIdOpInterface
    : public ProcessorIDInterface::ExternalModel<
          WorkgroupIdOpInterface, IREE::HAL::InterfaceWorkgroupIDOp> {
  unsigned getDimIndex(Operation *op) const {
    return cast<IREE::HAL::InterfaceWorkgroupIDOp>(op)
        .dimension()
        .getZExtValue();
  }
};

struct WorkgroupCountOpInterface
    : public ProcessorCountInterface::ExternalModel<
          WorkgroupCountOpInterface, IREE::HAL::InterfaceWorkgroupIDOp> {
  unsigned getDimIndex(Operation *op) const {
    return cast<IREE::HAL::InterfaceWorkgroupCountOp>(op)
        .dimension()
        .getZExtValue();
  }
};

struct WorkgroupTileSizeOpInterface
    : public ProcessorTileSizeInterface::ExternalModel<
          WorkgroupTileSizeOpInterface, IREE::HAL::InterfaceWorkgroupIDOp> {
  unsigned getDimIndex(Operation *op) const {
    return cast<IREE::HAL::InterfaceWorkgroupSizeOp>(op)
        .dimension()
        .getZExtValue();
  }
};

void registerProcessorOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<gpu::ThreadIdOp, ThreadIdOpInterface>();
  registry.addOpInterface<gpu::BlockDimOp, BlockDimOpInterface>();

  registry.addOpInterface<IREE::HAL::InterfaceWorkgroupIDOp,
                          WorkgroupIdOpInterface>();
  registry.addOpInterface<IREE::HAL::InterfaceWorkgroupCountOp,
                          WorkgroupCountOpInterface>();
  registry.addOpInterface<IREE::HAL::InterfaceWorkgroupSizeOp,
                          WorkgroupTileSizeOpInterface>();
}

}  // namespace iree_compiler
}  // namespace mlir
