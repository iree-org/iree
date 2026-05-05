// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_CPUENCODINGEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_CPUENCODINGEXTERNALMODELS_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::iree_compiler::IREE::CPU {

void registerCPUEncodingExternalModels(DialectRegistry &registry);

/// Returns x86 `MMAIntrinsic` cases whose required ISA extensions are all
/// present in `config` (`cpu_features` / target features) and whose tile
/// element types match `elementTypes` (LHS, RHS, ACC).
///
/// The data-tiling cost model uses this to enumerate candidates, and dispatch
/// creation uses an empty result as the early opt-out signal: an
/// `iree_codegen.inner_tiled` for this matmul on this target would have no
/// candidate intrinsic, so leave the matmul unannotated rather than produce
/// an untranslatable inner_tiled.
SmallVector<IREE::CPU::MMAIntrinsic>
matchMmaIntrinsics(MLIRContext *ctx, DictionaryAttr config,
                   ArrayRef<Type> elementTypes);

} // namespace mlir::iree_compiler::IREE::CPU

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_CPUENCODINGEXTERNALMODELS_H_
