// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#define DEBUG_TYPE "iree-gpu-encoding-external-models"

namespace mlir::iree_compiler::IREE::CPU {

using Codegen::MaterializeEncodingInfo;
using Codegen::TileMxNxK;

namespace {

//===----------------------------------------------------------------------===//
// Interface methods implementaion for iree_cpu.vmvx_encoding_layout.
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from when no specific architecture is
// targeted. For narrow-{M,N} cases, this only enumerates on narrow M. The
// narrow-N cases are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTilesVMVX(linalg::ContractionDimensions cDims,
                         IREE::Encoding::EncodingAttr encoding,
                         DictionaryAttr config) {
  bool hasUkernelSupport = hasUkernel(config);

  // TODO(hanchung): The ukernel path does not support 3d
  // codegen.query_tile_sizes op, so we disable dynamic tile shapes for
  // batch_matmul. Also, they are not set up for narrow M/N matmul, so it is
  // disabled when it is the case.
  if (!cDims.batch.empty() || getMatmulNarrowDim(encoding)) {
    hasUkernelSupport = false;
  }
  if (hasUkernelSupport) {
    // VMVX+ukernel uses dynamic tile shapes.
    return {TileMxNxK{ShapedType::kDynamic, ShapedType::kDynamic,
                      ShapedType::kDynamic}};
  }

  return {
      TileMxNxK{8, 8, 4}, // Some vaguely reasonable tile shape.
      TileMxNxK{4, 8, 4}, // Truncation of the above.
      TileMxNxK{2, 8, 4}, // Truncation of the above.
      TileMxNxK{1, 8, 4}, // Truncation of the above.
  };
}

struct VMVXDeviceEncodingLayoutAttrInterface
    : public Codegen::LayoutAttrInterface::ExternalModel<
          VMVXDeviceEncodingLayoutAttrInterface, VMVXEncodingLayoutAttr> {
  MaterializeEncodingInfo getEncodingInfo(Attribute attr,
                                          RankedTensorType type) const {
    auto layoutAttr = cast<VMVXEncodingLayoutAttr>(attr);
    auto encoding = llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
        type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    SmallVector<TileMxNxK> enumeratedTileMxNxK =
        enumerateMatmulTilesVMVX(encoding, layoutAttr.getConfiguration());
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getMatmulNarrowDim(encoding);
    // Choose a final matmul TileMxNxK from the above-enumarated tile shapes,
    // taking narrow dimensions into account.
    TileMxNxK chosenTileMxNxK = chooseMatmulTile(
        enumeratedTileMxNxK, narrowDim, encoding.getRoundDimsToArray());
    return getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
  }

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<VMVXEncodingLayoutAttr>(attr);
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    return lowerContractionOpToMultiMmaOp(b, linalgOp, convertedOperands,
                                          layoutAttr.getTargetAttr());
  }
};

} // namespace

void registerCPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::CPU::IREECPUDialect *dialect) {
        IREE::CPU::VMVXEncodingLayoutAttr::attachInterface<
            VMVXDeviceEncodingLayoutAttrInterface>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::CPU
