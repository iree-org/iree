// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

//===----------------------------------------------------------------------===//
// iree_cpu.encoding_solver
//===----------------------------------------------------------------------===//

static OpFoldResult mulAll(OpBuilder &builder, Location &loc,
                           ArrayRef<OpFoldResult> shape) {
  OpFoldResult res = builder.getIndexAttr(1);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  for (auto dimSize : shape) {
    res = affine::makeComposedFoldedAffineApply(builder, loc, d0 * d1,
                                                {res, dimSize});
  }
  return res;
}

OpFoldResult VMVXEncodingSolverAttr::calculateStorageElementCountInBytes(
    OpBuilder &builder, RankedTensorType type, ValueRange dynamicDims) const {
  auto encoding =
      llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(type.getEncoding());

  Location loc = builder.getUnknownLoc();
  SmallVector<OpFoldResult> shape =
      getMixedValues(type.getShape(), dynamicDims, builder);
  if (!encoding) {
    return mulAll(builder, loc, shape);
  }
  // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return mulAll(builder, loc, shape);
  }

  bool hasUkernelSupport = false;
  if (getTargetConfiguration().get(builder.getStringAttr("ukernels"))) {
    hasUkernelSupport = true;
  }

  if (hasUkernelSupport) {
    AffineExpr expr = builder.getAffineDimExpr(0);
    auto padTo16 = [&](int64_t dim) -> void {
      std::optional<unsigned> maybeMappedDim =
          encoding.mapDimToOperandIndex(dim);
      if (!maybeMappedDim) {
        return;
      }
      unsigned mappedDim = maybeMappedDim.value();
      shape[mappedDim] = affine::makeComposedFoldedAffineApply(
          builder, loc, expr.ceilDiv(16) * 16, {shape[mappedDim]});
    };
    for (auto m : cDims->m) {
      padTo16(m);
    }
    for (auto n : cDims->n) {
      padTo16(n);
    }
    for (auto k : cDims->k) {
      padTo16(k);
    }
  }

  return mulAll(builder, loc, shape);
}

Encoding::EncodingSolverInterfaceAttr
VMVXEncodingSolverAttr::cloneWithConfig(DictionaryAttr attr) const {
  return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(
      VMVXEncodingSolverAttr::get(getContext(), attr));
}

DictionaryAttr VMVXEncodingSolverAttr::getConfig() const {
  return getTargetConfiguration();
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
