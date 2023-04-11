// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGINFO_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGINFO_H_

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Utils/EncodingInfo.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir {
namespace iree_compiler {

enum class MatmulOperandRole {
  LHS,
  RHS,
  RESULT,
};

struct MatmulTileParams {
  int64_t M = 1;
  int64_t K = 1;
  int64_t N = 1;
};

/// Extracts encoding from the `tensorType` if specified.
std::optional<IREE::LinalgExt::TensorEncoding> getEncoding(
    RankedTensorType tensorType);

std::optional<MatmulType> getMatmulType(
    IREE::LinalgExt::TensorEncoding encoding);

std::optional<MatmulOperandRole> getMatmulOperandRole(
    IREE::LinalgExt::TensorEncoding encoding);

void adjustTileSizesToNarrowStaticShape(
    IREE::LinalgExt::MaterializeEncodingInfo &encodingInfo,
    ArrayRef<int64_t> shape);

IREE::LinalgExt::MaterializeEncodingInfo chooseEncodingInfoForMatmul(
    MatmulType type, MatmulOperandRole operandRole,
    MatmulTileParams tileParams);

IREE::LinalgExt::MaterializeEncodingValueFn getMaterializeEncodingValueFn(
    IREE::HAL::ExecutableTargetAttr targetAttr);

void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns,
    IREE::LinalgExt::MaterializeEncodingConversionTarget &target,
    IREE::LinalgExt::MaterializeEncodingTypeConverter &typeConverter,
    IREE::LinalgExt::MaterializeEncodingValueFn materializeEncodingValueFn);

// TODO(hanchung): Move the method to VMVX/EncodingInfo.h. This is required by
// TileAndDistributeToWorkgroupPass and VMVXMaterializeEncodingPass. It can not
// be in VMVX/EncodingInfo.h because there is a circular dependency. The Common/
// should not depend on other target backends.
FailureOr<IREE::LinalgExt::MaterializeEncodingValueInfo>
chooseDynamicEncodingInfoVMVXMicrokernels(RankedTensorType tensorType,
                                          OpBuilder &builder, Location loc);

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGINFO_H_
