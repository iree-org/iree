// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "iree-cpu-attrs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::CPU {

using Codegen::MaterializeEncodingInfo;
using Codegen::TileMxNxK;

//===----------------------------------------------------------------------===//
// iree_cpu.vmvx_encoding_layout
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from when no specific architecture is
// targeted. For narrow-{M,N} cases, this only enumerates on narrow M. The
// narrow-N cases are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTilesVMVX(IREE::Encoding::EncodingAttr encoding,
                         DictionaryAttr target) {
  SmallVector<TileMxNxK> nonUkernelCandidates = {
      TileMxNxK{8, 8, 4}, // Some vaguely reasonable tile shape.
      TileMxNxK{4, 8, 4}, // Truncation of the above.
      TileMxNxK{2, 8, 4}, // Truncation of the above.
      TileMxNxK{1, 8, 4}, // Truncation of the above.
  };
  if (!target) {
    return nonUkernelCandidates;
  }
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims)) {
    return nonUkernelCandidates;
  }

  auto ukernelAttr = target.getNamed("ukernels");
  bool hasUkernelSupport = false;
  if (ukernelAttr) {
    auto strAttr = dyn_cast<StringAttr>(ukernelAttr->getValue());
    if (strAttr && strAttr.getValue() == "all") {
      hasUkernelSupport = true;
    }
  }

  // TODO(hanchung): The ukernel path does not support 3d
  // codegen.query_tile_sizes op, so we disable dynamic tile shapes for
  // batch_matmul. Also, they are not set up for narrow M/N matmul, so it is
  // disabled when it is the case.
  if (!cDims->batch.empty() || getMatmulNarrowDim(encoding)) {
    hasUkernelSupport = false;
  }
  if (hasUkernelSupport) {
    // VMVX+ukernel uses dynamic tile shapes.
    return {TileMxNxK{ShapedType::kDynamic, ShapedType::kDynamic,
                      ShapedType::kDynamic}};
  }

  return nonUkernelCandidates;
}

MaterializeEncodingInfo
VMVXEncodingLayoutAttr::getEncodingInfo(RankedTensorType type) const {
  auto encoding =
      llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(type.getEncoding());

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
      enumerateMatmulTilesVMVX(encoding, getConfiguration());
  if (enumeratedTileMxNxK.empty()) {
    return info;
  }
  auto narrowDim = IREE::Encoding::getMatmulNarrowDim(encoding);
  // Choose a final matmul TileMxNxK from the above-enumarated tile shapes,
  // taking narrow dimensions into account.
  TileMxNxK chosenTileMxNxK = chooseMatmulTile(enumeratedTileMxNxK, narrowDim,
                                               encoding.getRoundDimsToArray());
  return getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
}

Operation *VMVXEncodingLayoutAttr::lowerOp(OpBuilder &b, Operation *op,
                                           TypeRange convertedResTypes,
                                           ValueRange convertedOperands) const {
  auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return nullptr;
  }

  auto resolver =
      [&](RankedTensorType type) -> FailureOr<MaterializeEncodingInfo> {
    return this->getEncodingInfo(type);
  };
  if (linalg::isaContractionOpInterface(linalgOp)) {
    FailureOr<Operation *> newOp = Codegen::lowerContractionOpWithEncoding(
        b, linalgOp, convertedOperands, /*transposeNarrowN=*/true, resolver);
    return newOp.value_or(nullptr);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREECPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::CPU
