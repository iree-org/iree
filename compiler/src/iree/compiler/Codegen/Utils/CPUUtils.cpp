// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-codegen-cpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::iree_compiler {

static const char kVscaleRangeAttrName[] = "vscale_range";

static llvm::cl::opt<bool> clEnableScalableVectorization(
    "iree-llvmcpu-enable-scalable-vectorization",
    llvm::cl::desc("Enable scalable vectorization if it is supported by the "
                   "target (e.g., +sve, +sve2 and/or +sme feature flags)"),
    llvm::cl::init(false));

// By default, IREE does not enable the Armv9-A streaming SVE mode in the
// presence of scalable vectors (even when using `+sme`), as currently there's
// no cost model of when it could be beneficial. This flag will effectively make
// IREE/LLVM switch from SVE to SSVE in dispatch regions with supported
// scalable vector operations.
static llvm::cl::opt<bool> clForceArmStreaming(
    "iree-llvmcpu-force-arm-streaming",
    llvm::cl::desc(
        "Enables Armv9-A streaming SVE mode for any dispatch region that "
        "contains supported scalable vector operations (i.e., use SSVE rather "
        "than SVE). Requires the +sme feature flag."),
    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<int> clVscaleFromUser(
    "iree-experimental-vscale-value",
    llvm::cl::desc(
        "The runtime value of vscale. This will _only_ be used for host-side "
        "code, e.g. to calculate storage sizes and workgroup counts. This is "
        "due to a current limitation of the host-side code not being able to "
        "properly query this value at runtime, see #21317 and #21590. Codegen "
        "will be vector-length agnostic and will be querying the value of "
        "vscale at runtime, as intended. For scalable vector code that "
        "propagates vscale ops into the host-side code, this value has to be "
        "explicitly set by the user, e.g. for SVE data-tiling on the AArch64 "
        "backend."),
    llvm::cl::Hidden, llvm::cl::init(-1));

FailureOr<Operation *> getRootOperation(ArrayRef<Operation *> computeOps) {
  Operation *rootOperation = nullptr;
  for (auto op : llvm::reverse(computeOps)) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not treat linalg ops that are all parallel as root operations in
      // this sweep.
      if (linalgOp.getNumLoops() == linalgOp.getNumParallelLoops()) {
        continue;
      }

      // All other linalg ops are root ops.
      rootOperation = op;
      break;
    }

    if (isa<TilingInterface>(op) &&
        !isa<tensor::PadOp, linalg::PackOp, linalg::UnPackOp,
             IREE::LinalgExt::MapLoadOp, IREE::LinalgExt::MapStoreOp>(op)) {
      // All other operations that implement this interface are root ops.
      rootOperation = op;
      break;
    }
  }

  if (!rootOperation) {
    // Check for elementwise operations.
    for (auto op : llvm::reverse(computeOps)) {
      if (isa<linalg::LinalgOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  if (!rootOperation) {
    // Check for relayout ops (pad/pack/unpack and the map_load/map_store
    // scatter/gather ops that encoding materialization folds into) by
    // themselves. These are excluded from the sweeps above so that a real
    // compute op in the same dispatch wins; a pure-relayout dispatch (e.g. a
    // `set_encoding` dispatch) still picks one of them here.
    for (auto op : llvm::reverse(computeOps)) {
      if (isa<tensor::PadOp, linalg::PackOp, linalg::UnPackOp,
              IREE::LinalgExt::MapLoadOp, IREE::LinalgExt::MapStoreOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  return rootOperation;
}

static const char kDecompositionAttrName[] = "enable_decomposition";
StringAttr getEnableDecompositionAttrName(MLIRContext *ctx) {
  return StringAttr::get(ctx, kDecompositionAttrName);
}
std::string getEnableDecompositionStr() { return kDecompositionAttrName; }

static const char kLoopPeelingAttrName[] = "enable_loop_peeling";
StringAttr getEnableLoopPeelingAttrName(MLIRContext *ctx) {
  return StringAttr::get(ctx, kLoopPeelingAttrName);
}
std::string getEnableLoopPeelingStr() { return kLoopPeelingAttrName; }

bool isOptEnabled(FunctionOpInterface funcOp, StringRef label) {
  DictionaryAttr config = getTranslationInfo(funcOp).getConfiguration();
  return config && config.contains(label);
}

bool isScalableVectorizationEnabled() { return clEnableScalableVectorization; }

bool isArmStreamingForced() { return clForceArmStreaming; }

unsigned getUserVscaleValue() {
  assert(clVscaleFromUser >= 1 && "Currently, vscale needs to be specified by "
                                  "the user for host-side code!");
  return clVscaleFromUser;
}

bool isProducerOfRootOp(Operation *op, Operation *rootOp) {
  if (!rootOp || op == rootOp) {
    return false;
  }
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user == rootOp) {
        return true;
      }
    }
  }
  return false;
}

bool hasAnySVEFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+sve") ||
         hasFeature(targetConfig, "+sve2") || hasFeature(targetConfig, "+v9a");
}

bool hasVFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+v");
}

bool hasZve32xFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+zve32x");
}

bool hasZve32fFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+zve32f");
}

bool hasZve64xFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+zve64x");
}

bool hasAnyVFeature(DictionaryAttr targetConfig) {
  return hasVFeature(targetConfig) || hasZve32xFeature(targetConfig) ||
         hasZve32fFeature(targetConfig) || hasZve64xFeature(targetConfig) ||
         hasFeature(targetConfig, "+zve64f") ||
         hasFeature(targetConfig, "+zve64d");
}

bool targetSupportsScalableVectors(DictionaryAttr targetConfig) {
  if (!targetConfig) {
    return false;
  }
  return (isAArch64(targetConfig) && hasAnySVEFeature(targetConfig)) ||
         (isRISCV(targetConfig) && hasAnyVFeature(targetConfig));
}

std::optional<std::pair<int64_t, int64_t>>
getConfigVscaleRange(DictionaryAttr targetConfig) {
  auto attr = targetConfig.getAs<ArrayAttr>(kVscaleRangeAttrName);
  if (!attr || attr.size() != 2) {
    return std::nullopt;
  }
  auto lo = dyn_cast<IntegerAttr>(attr[0]);
  auto hi = dyn_cast<IntegerAttr>(attr[1]);
  if (!lo || !hi) {
    return std::nullopt;
  }
  return std::make_pair(lo.getInt(), hi.getInt());
}
void addConfigVscaleRange(MLIRContext *context, int64_t vscaleMin,
                          int64_t vscaleMax,
                          SmallVectorImpl<NamedAttribute> &config) {
  auto i64 = IntegerType::get(context, 64);
  config.emplace_back(
      StringAttr::get(context, kVscaleRangeAttrName),
      ArrayAttr::get(context, {IntegerAttr::get(i64, vscaleMin),
                               IntegerAttr::get(i64, vscaleMax)}));
}

std::optional<vector::VscaleRange>
getDefaultVscaleRange(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (targetAttr) {
    DictionaryAttr targetConfig = targetAttr.getConfiguration();
    if (isAArch64(targetConfig) && hasAnySVEFeature(targetConfig)) {
      // For Arm SVE/SVE2 the scalable vector length is between 128-bit and
      // 2048-bit, corresponding to a vscale range of 1 to 16. See:
      // https://developer.arm.com/Architectures/Scalable%20Vector%20Extensions
      return vector::VscaleRange{1, 16};
    }
  }
  // TODO: Implement for other architectures.
  return std::nullopt;
}

std::optional<vector::VscaleRange>
getVscaleRange(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (!targetAttr) {
    return std::nullopt;
  }
  if (auto range = getConfigVscaleRange(targetAttr.getConfiguration())) {
    return vector::VscaleRange{static_cast<unsigned>(range->first),
                               static_cast<unsigned>(range->second)};
  }
  return getDefaultVscaleRange(targetAttr);
}

} // namespace mlir::iree_compiler
