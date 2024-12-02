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
// iree_cpu.cpu_encoding_layout
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from on riscv32.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv32([[maybe_unused]] DictionaryAttr config) {
  return {
      TileMxNxK{8, 8, 4}, // Some reasonable tile shape.
      TileMxNxK{4, 8, 4}, // Truncation of the above.
      TileMxNxK{2, 8, 4}, // Truncation of the above.
      TileMxNxK{1, 8, 4}, // Truncation of the above.
  };
}

// Enumerate tile sizes to choose from on arm64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileArm64(TypeRange elementTypes,
                                                       DictionaryAttr config) {
  // Data-tiling for SVE is not implemented yet.
  if (hasFeature(config, "+sve") || hasFeature(config, "+sve2")) {
    return {};
  }

  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32()) &&
        hasFeature(config, "+bf16")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use BFMMLA.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16, as we could typically have a 2x wider
      // tile in that case. However, on current CPUs, the existing tiles seem
      // wide enough already to approach peak performance.
      return {
          TileMxNxK{8, 8, 1}, // Aim to use FMLA or FMLAL.
          TileMxNxK{4, 8, 1}, // Truncation of the above.
          TileMxNxK{2, 8, 1}, // Truncation of the above.
          TileMxNxK{1, 8, 1}, // Truncation of the above.
      };
    }
  }

  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{8, 8, 8}, // Aim to use SMMLA.
          TileMxNxK{4, 8, 8}, // Truncation of the above.
          TileMxNxK{2, 8, 8}, // Truncation of the above.
          TileMxNxK{1, 8, 8}, // Truncation of the above.
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use SDOT.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
  }

  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(4) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{4, 8, 16},
          TileMxNxK{2, 8, 16},
          TileMxNxK{1, 8, 16},
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 8},
          TileMxNxK{4, 8, 8},
          TileMxNxK{2, 8, 8},
          TileMxNxK{1, 8, 8},
      };
    }
    return {
        TileMxNxK{4, 16, 2},
        TileMxNxK{2, 16, 2},
        TileMxNxK{1, 16, 2},
    };
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on x86-64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileX86_64(TypeRange elementTypes,
                                                        DictionaryAttr config) {
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32())) {
      if (hasFeature(config, "+avx512bf16")) {
        return {
            TileMxNxK{16, 16, 2}, // Aim to use VDPBF16PS (zmm).
            TileMxNxK{8, 16, 2},  // Truncation of the above.
            TileMxNxK{4, 16, 2},  // Truncation of the above.
            TileMxNxK{2, 16, 2},  // Truncation of the above.
            TileMxNxK{1, 16, 2},  // Truncation of the above.
        };
      }
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16.
      if (hasFeature(config, "+avx512f")) {
        return {
            TileMxNxK{16, 16, 1}, // Aim to use VFMADD* (zmm).
            TileMxNxK{8, 16, 1},  // Truncation of the above.
            TileMxNxK{4, 16, 1},  // Truncation of the above.
            TileMxNxK{2, 16, 1},  // Truncation of the above.
            TileMxNxK{1, 16, 1},  // Truncation of the above.
        };
      }
      if (hasFeature(config, "+avx")) {
        // Note: for good performance, most +avx users will also want to add
        // +fma, but that's a local instruction selection detail and the tile
        // layout is unaffected, as there are enough registers even with the
        // need for intermediate product registers when +fma is not used.
        return {
            TileMxNxK{8, 8, 1}, // Aim to use VFMADD* (ymm).
            TileMxNxK{4, 8, 1}, // Truncation of the above.
            TileMxNxK{2, 8, 1}, // Truncation of the above.
            TileMxNxK{1, 8, 1}, // Truncation of the above.
        };
      }
      // SSE fallback.
      return {
          TileMxNxK{8, 4, 1}, // Aim to use MULPS/ADDPS (xmm).
          TileMxNxK{4, 4, 1}, // Truncation of the above.
          TileMxNxK{2, 4, 1}, // Truncation of the above.
          TileMxNxK{1, 4, 1}, // Truncation of the above.
      };
    }
  }

  if (out.isSignlessInteger(32) &&
      ((lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8)) ||
       (lhs.isSignlessInteger(16) && rhs.isSignlessInteger(16)))) {
    if (hasFeature(config, "+avx512vnni")) {
      // This is the same tile size as with VPMADDWD as the only difference
      // is that VPDPWSSD accumulates. VPDPBUSD would call for {16, 16, 4} but
      // we can't easily use it because of its unsigned*signed semantics.
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPDPWSSD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx512bw")) {
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPMADDWD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx2")) {
      return {
          TileMxNxK{8, 8, 2}, // Aim to use VPMADDWD (ymm).
          TileMxNxK{4, 8, 2}, // Truncation of the above.
          TileMxNxK{2, 8, 2}, // Truncation of the above.
          TileMxNxK{1, 8, 2}, // Truncation of the above.
      };
    }
    // SSE fallback.
    return {
        TileMxNxK{8, 4, 2}, // Aim to use PMADDWD (xmm).
        TileMxNxK{4, 4, 2}, // Truncation of the above.
        TileMxNxK{2, 4, 2}, // Truncation of the above.
        TileMxNxK{1, 4, 2}, // Truncation of the above.
    };
  }

  if (out.isSignlessInteger(32) && lhs.isSignlessInteger(16) &&
      rhs.isUnsignedInteger(4)) {
    // Experimental s16u4s32 case. Focusing only on the vecmat case for now.
    if (hasFeature(config, "+avx512vnni")) {
      return {
          TileMxNxK{1, 32, 8}, // Aim to use VPDPBUSD (zmm).
      };
    }
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

static SmallVector<TileMxNxK>
enumerateMatmulTilesCPU(IREE::Encoding::EncodingAttr encoding,
                        DictionaryAttr config) {
  // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return {};
  }
  SmallVector<Type> elementTypes = encoding.getElementTypesArray();
  if (isAArch64(config)) {
    return enumerateMatmulTileArm64(elementTypes, config);
  }
  if (isX86_64(config)) {
    return enumerateMatmulTileX86_64(elementTypes, config);
  }
  if (isRISCV32(config)) {
    return enumerateMatmulTileRiscv32(config);
  }
  return {};
}

MaterializeEncodingInfo
CPUEncodingLayoutAttr::getEncodingInfo(RankedTensorType type) const {
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
      enumerateMatmulTilesCPU(encoding, getConfiguration());
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

Operation *CPUEncodingLayoutAttr::lowerOp(OpBuilder &b, Operation *op,
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
