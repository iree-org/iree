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

Attribute TensorUKernelProviderAttr::getDataLayoutForUKernel(
    Attribute encoding, DictionaryAttr targetConfiguration) const {
  auto encodingAttr =
      dyn_cast_if_present<IREE::Encoding::EncodingAttr>(encoding);
  if (!encodingAttr) {
    return {};
  }
  IREE::GPU::TargetAttr targetAttr = getGPUTargetAttr(targetConfiguration);
  if (!targetAttr || targetAttr.getArch() != "gfx942") {
    return {};
  }
  ArrayAttr indexingMapsAttr = encodingAttr.getUserIndexingMaps();
  if (!indexingMapsAttr) {
    return {};
  }
  if (failed(linalg::inferContractionDims(encodingAttr.getRootMaps()))) {
    return {};
  }
  SmallVector<Type> types = encodingAttr.getElementTypesArray();
  SmallVector<int64_t> iterationSizes = encodingAttr.getIterationSizesArray();
  if (types.size() != 3 || iterationSizes.size() != 3) {
    return {};
  }
  // Match the layouts based on UKernels implementation:
  // https://github.com/iree-org/iree/tree/main/compiler/plugins/target/ROCM/builtins/mlir_ukernel
  Type f16 = Float16Type::get(encoding.getContext());
  Type f32 = Float32Type::get(encoding.getContext());
  Type f8E4M3FNUZ = Float8E4M3FNUZType::get(encoding.getContext());
  if (types[0] == f16 && types[1] == f16 && types[2] == f32) {
    // UKernel: pingpong_dt_large_f16.
    return IREE::GPU::DataTiledMMAAttr::get(
        encoding.getContext(), IREE::GPU::MMAIntrinsic::MFMA_F32_16x16x16_F16,
        8, 2, 4, 4, 1);
  }
  if (types[0] == f8E4M3FNUZ && types[1] == f8E4M3FNUZ && types[2] == f32) {
    /// TODO(#21865): Remove the upper bound (8192) once the scratch memory
    /// issue is resolved.
    if (iterationSizes[1] >= 2048 && iterationSizes[1] <= 8192) {
      // UKernel: pingpong_dt_large_f8E4M3FNUZ.
      return IREE::GPU::DataTiledMMAAttr::get(
          encoding.getContext(),
          IREE::GPU::MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ, 8, 2, 4, 4, 1);
    } else {
      // UKernel: pingpong_dt_medium_f8E4M3FNUZ.
      return IREE::GPU::DataTiledMMAAttr::get(
          encoding.getContext(),
          IREE::GPU::MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ, 8, 1, 2, 8, 2);
    }
  }
  return {};
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
