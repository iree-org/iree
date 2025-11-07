// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.h"
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMUkernelBitcodeSupport.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::ROCM {

//===----------------------------------------------------------------------===//
// BuiltinTuningModuleAttr
//===----------------------------------------------------------------------===//

FailureOr<mlir::ModuleOp>
BuiltinTuningModuleAttr::getModule(Operation * /*annotationSite*/) const {
  auto &rocmDialect = cast<ROCMDialect>(getDialect());
  return rocmDialect.getOrLoadBuiltinModule(getBuiltinFilename());
}

//===----------------------------------------------------------------------===//
// UKernelProviderAttr
//===----------------------------------------------------------------------===//

std::optional<LogicalResult> UKernelProviderAttr::createAndReplaceWithUkernelOp(
    RewriterBase &rewriter, StringRef name, DictionaryAttr targetConfiguration,
    Operation *contextualOp, ArrayRef<Value> inputs, ArrayRef<Value> outputs,
    SmallVectorImpl<Value> &otherOperands) const {
  if (name.contains("argmax")) {
    return handleArgmaxUkernel(rewriter, name, targetConfiguration,
                               contextualOp, inputs, outputs, otherOperands);
  } else if (name.contains("multi_mma_mfma")) {
    return handleInnerTiledMmaUkernel(rewriter, name, targetConfiguration,
                                      contextualOp, inputs, outputs,
                                      otherOperands);
  }
  return std::nullopt;
}

//===---------------------------------------------------------------------===//
// rocm.tensor_ukernel_provider
//===---------------------------------------------------------------------===//

FailureOr<Operation *>
TensorUKernelProviderAttr::getMLIRUKernel(StringRef name, DictionaryAttr,
                                          Operation *annotationSite) const {
  auto *symbolTableOp = SymbolTable::getNearestSymbolTable(annotationSite);
  SymbolTable symbolTable(symbolTableOp);
  return symbolTable.lookup(name);
}

// Helper for getDataLayoutForUKernel: returns true if the actual
// `iterationSizes` satisfy any `iterationSizeConstraints`.
static bool checkIterationSizeConstraints(ArrayRef<int64_t> iterationSizes,
                                          ArrayAttr iterationSizeConstraints) {
  for (Attribute c : iterationSizeConstraints) {
    auto constraint = dyn_cast<UKernelIterationSizeConstraintAttr>(c);
    if (!constraint) {
      return false;
    }
    IntegerAttr index = constraint.getIndex();
    if (!index) {
      return false;
    }
    int64_t indexVal = index.getInt();
    if (indexVal < 0 || indexVal >= iterationSizes.size()) {
      return false;
    }
    // For now, assume a dynamic dimension is very large and any division
    // constraint is satisfied to keep the performance state on current models
    // (llama) as is.
    // TODO(#22370): This is not ideal and can be improved once we support value
    // bounds on dynamic dimensions for encodings.
    if (IntegerAttr sizeMin = constraint.getSizeMin()) {
      if (ShapedType::isStatic(iterationSizes[indexVal]) &&
          iterationSizes[indexVal] < sizeMin.getInt()) {
        return false;
      }
    }
    if (IntegerAttr sizeMax = constraint.getSizeMax()) {
      if (ShapedType::isDynamic(iterationSizes[indexVal])) {
        return false;
      }
      if (iterationSizes[indexVal] > sizeMax.getInt()) {
        return false;
      }
    }
    if (IntegerAttr sizeDiv = constraint.getSizeDiv()) {
      if (sizeDiv.getInt() <= 0) {
        return false;
      }
      if (ShapedType::isDynamic(iterationSizes[indexVal])) {
        return true;
      }
      if (iterationSizes[indexVal] % sizeDiv.getInt()) {
        return false;
      }
    }
  }
  return true;
}

Attribute TensorUKernelProviderAttr::getDataLayoutForUKernel(
    Attribute encoding, DictionaryAttr targetConfiguration) const {
  auto encodingAttr =
      dyn_cast_if_present<IREE::Encoding::EncodingAttr>(encoding);
  if (!encodingAttr) {
    return {};
  }
  IREE::GPU::TargetAttr targetAttr = getGPUTargetAttr(targetConfiguration);
  IREE::Encoding::EncodingOpType opType = encodingAttr.getOpType().getValue();
  if (opType != IREE::Encoding::EncodingOpType::matmul &&
      opType != IREE::Encoding::EncodingOpType::scaled_matmul) {
    return {};
  }
  SmallVector<Type> types = encodingAttr.getElementTypesArray();
  SmallVector<int64_t> iterationSizes = encodingAttr.getIterationSizesArray();
  // Matmul has LHS, RHS, and ACC types. Scaled matmul has LHS, RHS, LHS scale,
  // RHS scale, and ACC types.
  int64_t expectedTypeSize =
      (opType == IREE::Encoding::EncodingOpType::matmul) ? 3 : 5;
  // Matmul has M, N, and K iteration sizes. Scaled matmul has M, N, K, and Kb
  // iteration sizes.
  int64_t expectedIterationSize =
      (opType == IREE::Encoding::EncodingOpType::matmul) ? 3 : 4;
  if (types.size() != expectedTypeSize ||
      iterationSizes.size() != expectedIterationSize) {
    return {};
  }
  auto &rocmDialect = cast<ROCMDialect>(getDialect());
  // Iterate over all MLIR ukernels and select the one with highest benefit.
  Attribute selectedMma;          // Initial value means no match.
  int64_t selectedMmaBenefit = 0; // Initial value not actually used.

  for (Util::FuncOp funcOp : rocmDialect.getMlirUKernels()) {
    // Require MLIR ukernels to have a ukernel_info attribute, otherwise we
    // won't be able to know how to data-tile for them.
    auto info =
        dyn_cast_if_present<UKernelInfoAttr>(funcOp->getAttr(kUKernelInfoName));
    if (!info) {
      continue;
    }
    // If a previously selected mma has a larger or equal benefit, skip.
    if (selectedMma && selectedMmaBenefit >= info.getBenefit()) {
      continue;
    }
    // Match the element types.
    auto matchTypes = dyn_cast_if_present<ArrayAttr>(
        info.getMatch().get(kUKernelInfoTypesName));
    auto actualTypes =
        ArrayAttr::get(matchTypes.getContext(),
                       llvm::map_to_vector(types, [](Type v) -> Attribute {
                         return TypeAttr::get(v);
                       }));
    if (matchTypes != actualTypes) {
      continue;
    }
    // Match any constraints on iteration sizes.
    if (auto iterationSizeConstraints = dyn_cast_if_present<ArrayAttr>(
            info.getMatch().get(kUKernelInfoIterationSizesConstraintsName))) {
      if (!checkIterationSizeConstraints(iterationSizes,
                                         iterationSizeConstraints)) {
        continue;
      }
    }
    // Read the data-tiled-layout attribute.
    Attribute mma = info.getMma();
    if (!mma) {
      continue;
    }
    // Depending on the type of data-tiled layout attribute, read the
    // appropriate kind of MMA-like intrinsic and check that it's supported by
    // the target.
    if (opType == IREE::Encoding::EncodingOpType::matmul) {
      auto dtMma = dyn_cast<GPU::DataTiledMMAAttr>(mma);
      if (!dtMma) {
        continue;
      }
      // Regular MMA intrinsic.
      auto intrinsicAttr =
          GPU::MMAAttr::get(matchTypes.getContext(), dtMma.getIntrinsic());
      if (!llvm::is_contained(targetAttr.getWgp().getMma(), intrinsicAttr)) {
        continue;
      }
    } else if (opType == IREE::Encoding::EncodingOpType::scaled_matmul) {
      auto dtScaledMma = dyn_cast<GPU::DataTiledScaledMMAAttr>(mma);
      if (!dtScaledMma) {
        continue;
      }
      // Scaled MMA intrinsic.
      auto intrinsicAttr = GPU::ScaledMMAAttr::get(
          matchTypes.getContext(), dtScaledMma.getIntrinsic(),
          /*lhs_elem_type=*/types[0], /*rhs_elem_type=*/types[1],
          /*acc_elem_type=*/types[4], /*col_major=*/false);
      if (!llvm::is_contained(targetAttr.getWgp().getScaledMma(),
                              intrinsicAttr)) {
        continue;
      }
    } else {
      // Unhandled type of data-tiled-layout attr.
      continue;
    }
    // Selected!
    selectedMma = mma;
    selectedMmaBenefit = info.getBenefit();
  }

  return selectedMma;
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void ROCMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::ROCM
