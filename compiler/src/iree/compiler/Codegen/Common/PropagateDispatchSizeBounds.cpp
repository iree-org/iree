// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATEDISPATCHSIZEBOUNDSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Fold statically-known block/grid dimensions to constants, since that's
// stronger than just imposing an upper bound on their size.
static void foldConstantBounds(
    FunctionOpInterface funcOp,
    const std::optional<SmallVector<int64_t>> &staticWorkgroupSizes,
    ArrayRef<int64_t> staticWorkgroupCounts,
    std::optional<int64_t> subgroupSize) {
  IRRewriter rewriter(funcOp->getContext());
  auto rewriteToConstant = [&](Operation *op, int64_t constant) {
    rewriter.setInsertionPoint(op);
    Type constType = op->getResult(0).getType();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, constType, IntegerAttr::get(constType, constant));
  };
  funcOp->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](gpu::BlockDimOp blockDimOp) {
          int32_t dim = static_cast<int32_t>(blockDimOp.getDimension());
          if (staticWorkgroupSizes.has_value() &&
              staticWorkgroupSizes->size() > dim) {
            rewriteToConstant(blockDimOp, (*staticWorkgroupSizes)[dim]);
          }
        })
        .Case([&](gpu::SubgroupSizeOp subgroupSizeOp) {
          if (subgroupSize.has_value()) {
            rewriteToConstant(subgroupSizeOp, *subgroupSize);
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupSizeOp wgSizeOp) {
          size_t dim = wgSizeOp.getDimension().getZExtValue();
          if (staticWorkgroupSizes.has_value() &&
              staticWorkgroupSizes->size() > dim) {
            rewriteToConstant(wgSizeOp, (*staticWorkgroupSizes)[dim]);
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupCountOp wgCountOp) {
          size_t dim = wgCountOp.getDimension().getZExtValue();
          int64_t size = staticWorkgroupCounts.size() > dim
                             ? staticWorkgroupCounts[dim]
                             : ShapedType::kDynamic;
          if (ShapedType::isStatic(size)) {
            rewriteToConstant(wgCountOp, size);
          }
        })
        .Default([](Operation *) {});
  });
}

static void applyBounds(FunctionOpInterface funcOp,
                        ArrayRef<std::optional<int64_t>> workgroupSizes,
                        ArrayRef<std::optional<int64_t>> workgroupCounts,
                        std::optional<int64_t> maxSubgroupSize,
                        std::optional<int64_t> subgroupIdBound) {
  Builder b(funcOp->getContext());
  funcOp->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](gpu::LaneIdOp laneIdOp) {
          if (maxSubgroupSize) {
            laneIdOp.setUpperBoundAttr(b.getIndexAttr(*maxSubgroupSize));
          }
        })
        .Case([&](gpu::SubgroupSizeOp subgroupSizeOp) {
          if (maxSubgroupSize) {
            subgroupSizeOp.setUpperBoundAttr(b.getIndexAttr(*maxSubgroupSize));
          }
        })
        .Case([&](gpu::SubgroupIdOp subgroupIdOp) {
          if (subgroupIdBound) {
            subgroupIdOp.setUpperBoundAttr(b.getIndexAttr(*subgroupIdBound));
          }
        })
        .Case([&](gpu::ThreadIdOp tidOp) {
          std::optional<int64_t> bound =
              workgroupSizes[static_cast<uint32_t>(tidOp.getDimension())];
          if (bound) {
            tidOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](gpu::BlockDimOp blockDimOp) {
          std::optional<int64_t> bound =
              workgroupSizes[static_cast<int32_t>(blockDimOp.getDimension())];
          if (bound) {
            blockDimOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupSizeOp wgSizeOp) {
          std::optional<int64_t> bound =
              workgroupSizes[wgSizeOp.getDimension().getZExtValue()];
          if (bound) {
            wgSizeOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupIDOp wgIdOp) {
          std::optional<int64_t> bound =
              workgroupCounts[wgIdOp.getDimension().getZExtValue()];
          if (bound) {
            wgIdOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupCountOp wgCountOp) {
          std::optional<int64_t> bound =
              workgroupCounts[wgCountOp.getDimension().getZExtValue()];
          if (bound) {
            wgCountOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Default([](Operation *) {});
  });
}

struct PropagateDispatchSizeBoundsPass final
    : impl::PropagateDispatchSizeBoundsPassBase<
          PropagateDispatchSizeBoundsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<std::optional<int64_t>, 3> workgroupSizes(3, std::nullopt);
    SmallVector<std::optional<int64_t>, 3> workgroupCounts(3, std::nullopt);

    IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(funcOp);
    if (gpuTarget) {
      ArrayRef<int32_t> targetWorkgroupSizes =
          gpuTarget.getWgp().getMaxWorkgroupSizes().asArrayRef();
      ArrayRef<int32_t> targetWorkgroupCounts =
          gpuTarget.getWgp().getMaxWorkgroupCounts().asArrayRef();
      llvm::transform(targetWorkgroupSizes, workgroupSizes.begin(),
                      [](int32_t x) { return std::optional<int64_t>{x}; });
      llvm::transform(targetWorkgroupCounts, workgroupCounts.begin(),
                      [](int32_t x) { return std::optional<int64_t>{x}; });
    }

    std::optional<SmallVector<int64_t>> staticWorkgroupSize =
        getWorkgroupSize(funcOp);

    // Check if a specific subgroup size has been explicitly chosen via the
    // codegen pipeline configuration.
    std::optional<int64_t> staticSubgroupSize = getSubgroupSize(funcOp);

    IREE::Codegen::DispatchConfigOp configOp;
    if (useDispatchConfig) {
      configOp = getDispatchConfigOp(funcOp);
      if (configOp) {
        if (std::optional<ArrayRef<int64_t>> wgSize =
                configOp.getWorkgroupSize()) {
          staticWorkgroupSize = llvm::to_vector(wgSize.value());
        }
        if (std::optional<uint64_t> sgSize = configOp.getSubgroupSize()) {
          staticSubgroupSize = static_cast<int64_t>(*sgSize);
        }
      }
    } else {
      // Late in codegen, we've reconciled the workgroup size onto the export
      // op.
      if (std::optional<IREE::HAL::ExecutableExportOp> exportOp =
              getEntryPoint(funcOp)) {
        if (std::optional<ArrayAttr> exportWorkgroupSize =
                exportOp->getWorkgroupSize()) {
          staticWorkgroupSize = llvm::map_to_vector(
              exportWorkgroupSize->getAsRange<IntegerAttr>(),
              [](IntegerAttr a) { return a.getInt(); });
        }

        if (std::optional<uint64_t> exportSubgroupSize =
                exportOp->getSubgroupSizeAsUInt()) {
          staticSubgroupSize = static_cast<int64_t>(*exportSubgroupSize);
        }
      }
    }

    // Determine min and max subgroup size bounds. When a specific subgroup
    // size has been picked, min == max == that size. Otherwise, use the
    // range from the GPU target's WGP info.
    std::optional<int64_t> minSubgroupSize;
    std::optional<int64_t> maxSubgroupSize;
    if (staticSubgroupSize) {
      minSubgroupSize = maxSubgroupSize = staticSubgroupSize;
    } else if (gpuTarget) {
      assert(!gpuTarget.getWgp().getSubgroupSizeChoices().empty() &&
             "GPU target must have at least one subgroup size choice");
      minSubgroupSize = gpuTarget.getMinSubgroupSize();
      maxSubgroupSize = gpuTarget.getMaxSubgroupSize();
      if (*minSubgroupSize == *maxSubgroupSize) {
        // There's only one option, so we know what it is.
        staticSubgroupSize = maxSubgroupSize;
      }
    }

    if (staticWorkgroupSize) {
      // Target info with no workgroup sizes gives a 0-length array, hence no
      // zip_equal.
      for (auto [size, staticSize] :
           llvm::zip(workgroupSizes, *staticWorkgroupSize)) {
        size = staticSize;
      }
    }
    SmallVector<int64_t> staticWorkgroupCounts;
    if (useDispatchConfig && configOp) {
      staticWorkgroupCounts = configOp.getStaticNumWorkgroups();
    } else {
      staticWorkgroupCounts = getStaticNumWorkgroups(funcOp);
    }
    assert(staticWorkgroupCounts.size() <= 3 &&
           "workgroup counts are 3D at most");
    for (auto [count, staticCount] :
         llvm::zip(workgroupCounts, staticWorkgroupCounts)) {
      if (staticCount != ShapedType::kDynamic) {
        count = staticCount;
      }
    }

    // Compute the subgroup ID bound: max total threads / min subgroup size.
    std::optional<int64_t> maxFlatWorkgroupSize;
    std::optional<int64_t> subgroupIdBound;
    if (staticWorkgroupSize) {
      maxFlatWorkgroupSize = llvm::product_of(*staticWorkgroupSize);
    }
    if (gpuTarget) {
      maxFlatWorkgroupSize = std::min(
          maxFlatWorkgroupSize.value_or(std::numeric_limits<int64_t>::max()),
          static_cast<int64_t>(
              gpuTarget.getWgp().getMaxThreadCountPerWorkgroup()));
    }
    if (maxFlatWorkgroupSize && minSubgroupSize) {
      subgroupIdBound =
          llvm::divideCeil(*maxFlatWorkgroupSize, *minSubgroupSize);
    }

    foldConstantBounds(funcOp, staticWorkgroupSize, staticWorkgroupCounts,
                       staticSubgroupSize);
    applyBounds(funcOp, workgroupSizes, workgroupCounts, maxSubgroupSize,
                subgroupIdBound);

    if (auto *gpuDialect = getContext().getLoadedDialect<gpu::GPUDialect>()) {
      if (staticWorkgroupSize && gpuTarget) {
        std::array<int32_t, 3> blockSize = {1, 1, 1};
        llvm::transform(ArrayRef<int64_t>{*staticWorkgroupSize}.take_front(3),
                        blockSize.begin(), llvm::StaticCastTo<int32_t>);
        gpuDialect->getKnownBlockSizeAttrHelper().setAttr(
            funcOp, DenseI32ArrayAttr::get(funcOp->getContext(), blockSize));
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
