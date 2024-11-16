// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::GPU {

// Struct describing the detailed subgroup-level layout of a MMA intrinsic.
// Together with element type information and subgroup size, it completes the
// full description of the semantics of a MMA intrinsic.
//
// Note: It is not possible to infer subgroup size from the information in this
// struct. The product of the `thread` sizes here is often, but not always equal
// to subgroup size. When the product of the `thread` sizes (call that product
// `P`) is smaller than subgroup size, it must be a divisor of it, and the
// semantics in that case are that threads within the subgroup whose thread-ids
// differ by a multiple of `P`, are accessing the same elements.
//
// Example observed in RDNA3 WMMA Wave32 intrinsics:
// If the subgroup size is 32 but the product `P` of `thread` sizes is 16, that
// means that each element is being accessed by 2 threads (2 = 32/16), and the
// threads accessing the same element are those whose tids are exactly 16 apart.
struct MMASingleSubgroupLayout {
  // Internal dimensions (as in TileSwizzle::Dim::Kind::Internal) that are
  // outer-most in the layout. This happens when a MMA op, seen on a single
  // thread, has an operand that consists of multiple elements, and these elems
  // are NOT contiguous.
  // This is not used by every MMA op; ops which don't use that simply have 1's.
  SmallVector<int64_t, 2> outer;
  // Cross-thread dimensions (as in TileSwizzle::Dim::Kind::CrossThread).
  // This is the kind of dimension that is present in all GPU MMA ops, by
  // definition of "SIMT". It is still possible for one of the `thread` dims to
  // be 1, but not both.
  SmallVector<int64_t, 2> thread;
  // Strides corresponding to the cross-thread dimensions.
  SmallVector<int64_t, 2> tstrides;
  // Internal dimensions (as in TileSwizzle::Dim::Kind::Internal) that are
  // inner-most in the layout. This happens when a MMA op, seen on a single
  // thread, has an operand that consists of multiple elements, and these elems
  // are contiguous.
  // This is not used by every MMA op; ops which don't use that simply have 1's.
  SmallVector<int64_t, 2> element;
};

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                MMAFragment fragment);

MMASingleSubgroupLayout getSingleSubgroupLayout(VirtualMMAIntrinsic intrinsic,
                                                MMAFragment fragment);

MMASingleSubgroupLayout getSingleSubgroupLayout(MmaInterfaceAttr mmaKind,
                                                MMAFragment fragment);

// Struct describing the shape of a MMA operation, but not the detailed layout.
// TODO(bjacob): the only user outside of IREEGPUAttrs.cpp is
// LLVMGPU/TransformExtensions, so maybe make that internal again if/when that
// goes away.
struct OpaqueMmaLayout {
  int64_t mSize = 0;
  int64_t nSize = 0;
  int64_t kSize = 0;
  Type aType;
  Type bType;
  Type cType;
};

OpaqueMmaLayout getOpaqueMMALayout(MLIRContext *context,
                                   IREE::GPU::MMAIntrinsic intrinsic);

} // namespace mlir::iree_compiler::IREE::GPU

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h.inc"
#undef GET_ATTRDEF_CLASSES
// clang-format on

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_
