// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "cpu-materialize-encoding"

namespace mlir::iree_compiler {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileMxNxK;

#define GEN_PASS_DEF_CPUMATERIALIZEDEVICEENCODINGPASS
#define GEN_PASS_DEF_CPUMATERIALIZEHOSTENCODINGPASS
#include "iree/compiler/Codegen/Common/CPU/Passes.h.inc"

// Enumerate tile sizes to choose from on riscv32.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in IREE::CPU::chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv32(IREE::HAL::ExecutableTargetAttr target) {
  if (hasUkernel(target)) {
    return {
        TileMxNxK{8, 8, 4}, // Some reasonable tile shape.
        TileMxNxK{4, 8, 4}, // Truncation of the above.
        TileMxNxK{2, 8, 4}, // Truncation of the above.
        TileMxNxK{1, 8, 4}, // Truncation of the above.
    };
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on arm64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in IREE::CPU::chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileArm64(TypeRange elementTypes,
                         IREE::HAL::ExecutableTargetAttr target) {
  // Data-tiling for SVE is not implemented yet.
  if (hasFeature(target, "+sve") || hasFeature(target, "+sve2")) {
    return {};
  }

  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32()) &&
        hasFeature(target, "+bf16")) {
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
    if (hasFeature(target, "+i8mm")) {
      return {
          TileMxNxK{8, 8, 8}, // Aim to use SMMLA.
          TileMxNxK{4, 8, 8}, // Truncation of the above.
          TileMxNxK{2, 8, 8}, // Truncation of the above.
          TileMxNxK{1, 8, 8}, // Truncation of the above.
      };
    }
    if (hasFeature(target, "+dotprod")) {
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
    if (hasFeature(target, "+i8mm")) {
      return {
          TileMxNxK{4, 8, 16},
          TileMxNxK{2, 8, 16},
          TileMxNxK{1, 8, 16},
      };
    }
    if (hasFeature(target, "+dotprod")) {
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
// are handled by transposition in IREE::CPU::chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileX86_64(TypeRange elementTypes,
                          IREE::HAL::ExecutableTargetAttr target) {
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32())) {
      if (hasFeature(target, "+avx512bf16")) {
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
      if (hasFeature(target, "+avx512f")) {
        return {
            TileMxNxK{16, 16, 1}, // Aim to use VFMADD* (zmm).
            TileMxNxK{8, 16, 1},  // Truncation of the above.
            TileMxNxK{4, 16, 1},  // Truncation of the above.
            TileMxNxK{2, 16, 1},  // Truncation of the above.
            TileMxNxK{1, 16, 1},  // Truncation of the above.
        };
      }
      if (hasFeature(target, "+avx")) {
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
    if (hasFeature(target, "+avx512vnni")) {
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
    if (hasFeature(target, "+avx512bw")) {
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPMADDWD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(target, "+avx2")) {
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
    if (hasFeature(target, "+avx512vnni")) {
      return {
          TileMxNxK{1, 32, 8}, // Aim to use VPDPBUSD (zmm).
      };
    }
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

static SmallVector<TileMxNxK>
enumerateMatmulTileMxNxK(IREE::Encoding::EncodingAttr encoding,
                         IREE::HAL::ExecutableTargetAttr target) {
  // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return {};
  }
  // Enumerate available tile shapes for the given encoding and target.
  SmallVector<Type> elementTypes = encoding.getElementTypesArray();
  if (isAArch64(target)) {
    return enumerateMatmulTileArm64(elementTypes, target);
  }
  if (isX86_64(target)) {
    return enumerateMatmulTileX86_64(elementTypes, target);
  }
  if (isRISCV32(target)) {
    return enumerateMatmulTileRiscv32(target);
  }
  return {};
}

static FailureOr<MaterializeEncodingInfo>
materializeEncodingForTarget(RankedTensorType tensorType,
                             IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto encoding =
      dyn_cast_or_null<IREE::Encoding::EncodingAttr>(tensorType.getEncoding());
  if (!encoding) {
    return failure();
  }

  SmallVector<TileMxNxK> enumeratedTileMxNxK =
      enumerateMatmulTileMxNxK(encoding, targetAttr);
  if (enumeratedTileMxNxK.empty()) {
    return failure();
  }
  auto narrowDim = IREE::Encoding::getMatmulNarrowDim(encoding);
  // Choose a final matmul TileMxNxK from the above-enumarated tile shapes,
  // taking narrow dimensions into account.
  TileMxNxK chosenTileMxNxK = IREE::CPU::chooseMatmulTile(
      enumeratedTileMxNxK, narrowDim, encoding.getRoundDimsToArray());

  // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
  // based on its operand index in the matmul.
  return IREE::Codegen::getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
}

static FailureOr<MaterializeEncodingValueInfo>
chooseDynamicEncodingInfoVMVXMicrokernels(RankedTensorType tensorType,
                                          OpBuilder &builder, Location loc) {
  SmallVector<Type> resultTypes(tensorType.getRank(), builder.getIndexType());
  auto op = builder.create<IREE::Codegen::QueryTileSizesOp>(
      loc, resultTypes, TypeAttr::get(tensorType));
  MaterializeEncodingValueInfo result;
  result.innerTileSizes = op.getResults();
  return result;
}

static MaterializeEncodingValueFn
getMaterializeEncodingValueFn(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isVMVXBackend(targetAttr) && hasUkernel(targetAttr)) {
    return chooseDynamicEncodingInfoVMVXMicrokernels;
  }
  return {};
}

static LogicalResult
materializeFuncOpEncodings(FunctionOpInterface funcOp,
                           IREE::HAL::ExecutableTargetAttr targetAttr) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet materializeEncodingPattern(ctx);
  // On CPU, we use transposeNarrowN=true for a combination of reasons:
  // 1. As linalg.matmul materializes into linalg.mmt4d, which has a transposed
  //    RHS and therefore LHS<->RHS symmetry, transposeNarrowN is easy to
  //    implement at that level.
  // 2. We use ukernels, and this allows writing 2x fewer narrow ukernels.
  // 3. Heuristics for cache-friendly dispatch tiling can get complex on CPU,
  //    so it is nice that they have fewer narrow cases to consider.
  IREE::Codegen::LayoutAttrInterface layoutAttr;
  if (isVMVXBackend(targetAttr)) {
    layoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(
        IREE::CPU::VMVXEncodingLayoutAttr::get(ctx,
                                               targetAttr.getConfiguration()));
  }
  MaterializeEncodingTypeConverter typeConverter(
      materializeEncodingForTarget, targetAttr, /*transposeNarrowN=*/true,
      layoutAttr);
  MaterializeEncodingConversionTarget target(*ctx);
  auto materializeEncodingValueFn = getMaterializeEncodingValueFn(targetAttr);
  populateMaterializeEncodingIntoPackUnPackPatterns(
      materializeEncodingPattern, typeConverter, materializeEncodingValueFn);
  populateShapeIndependentMaterializeEncodingPatterns(
      materializeEncodingPattern, target, typeConverter,
      materializeEncodingValueFn);

  if (failed(applyPartialConversion(funcOp, target,
                                    std::move(materializeEncodingPattern)))) {
    funcOp.emitOpError("materialization failed");
    return failure();
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and
  // resolve dims ops.
  {
    RewritePatternSet patterns(ctx);
    tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("folding patterns failed");
      return failure();
    }
  }

  return success();
}

// Returns the executable targets used within |funcOp|.
//
// TODO(multi-device): delete this pass and rely on tensor-based analysis to
// materialize encodings based on where tensors are used. This pass is not able
// to handle that.
static std::optional<SetVector<IREE::HAL::ExecutableTargetAttr>>
getFuncExecutableTargetAttrs(FunctionOpInterface funcOp,
                             IREE::Stream::AffinityAnalysis &affinityAnalysis,
                             IREE::HAL::DeviceAnalysis &deviceAnalysis) {
  // Get a set of all unique affinities used by resources within the function.
  SetVector<IREE::Stream::AffinityAttr> uniqueAffinityAttrs;
  SmallVector<IREE::Stream::AffinityAttr> lookupAffinityAttrs;
  funcOp.walk([&](Operation *op) {
    if (affinityAnalysis.tryLookupExecutionAffinity(op, lookupAffinityAttrs)) {
      uniqueAffinityAttrs.insert(lookupAffinityAttrs.begin(),
                                 lookupAffinityAttrs.end());
    }
    lookupAffinityAttrs.clear();
  });

  // Resolve affinities to executable targets.
  SetVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  for (auto affinityAttr : uniqueAffinityAttrs) {
    deviceAnalysis.gatherRequiredExecutableTargets(affinityAttr, funcOp,
                                                   executableTargetAttrs);
  }
  return executableTargetAttrs;
}

struct CPUMaterializeHostEncodingPass
    : public impl::CPUMaterializeHostEncodingPassBase<
          CPUMaterializeHostEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, tensor::TensorDialect,
                IREE::Codegen::IREECodegenDialect, IREE::CPU::IREECPUDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Run required analysis passes.
    IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      return signalPassFailure();
    }
    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      // Gather the required executable targets for the function. Note that it's
      // possible there are more required for ops nested within the function but
      // this pass is a hack and can't handle that :shrug:.
      auto executableTargets = getFuncExecutableTargetAttrs(
          funcOp, affinityAnalysis, deviceAnalysis);
      if (!executableTargets) {
        funcOp.emitOpError()
            << "could not determine executable targets for the function";
        return signalPassFailure();
      } else if (executableTargets->empty()) {
        // Probably no tensors.
        continue;
      }

      // HACK: this pass is run on the host _but shouldn't be_. Because it's
      // run on the host and IREE is a compiler capable of multi-targeting there
      // may be multiple executable targets at any point in the host program.
      // This pass can't handle that and assumes it's been checked earlier by
      // spooky action at a distance. This needs to be fixed.
      if (executableTargets->size() != 1) {
        funcOp.emitOpError() << "has multiple executable targets and CPU data "
                                "tiling isn't built to support that";
        return signalPassFailure();
      }

      // Materialize encodings within the function.
      if (failed(
              materializeFuncOpEncodings(funcOp, executableTargets->front()))) {
        return signalPassFailure();
      }
    }
  }
};

// NOTE: this runs on host modules and executables and has two paths to handle
// that. It should _not_ be running on both - target-specific codegen passes
// are not allowed on host programs and it's a big violation of layering that
// this exists.
struct CPUMaterializeDeviceEncodingPass
    : public impl::CPUMaterializeDeviceEncodingPassBase<
          CPUMaterializeDeviceEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, tensor::TensorDialect,
                IREE::Codegen::IREECodegenDialect, IREE::CPU::IREECPUDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto executableTargetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (failed(materializeFuncOpEncodings(funcOp, executableTargetAttr))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler
