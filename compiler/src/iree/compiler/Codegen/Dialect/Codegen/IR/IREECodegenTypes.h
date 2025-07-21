// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IR_IREECODEGENTYPES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IR_IREECODEGENTYPES_H_

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler::IREE::Codegen {
//===----------------------------------------------------------------------===//
// Layout Struct Types.
//===----------------------------------------------------------------------===//

// Metadata for a swizzle, that is, an (expand_shape -> transposition)
// pair of ops performing a change of layout within the tiles. This is used
// on GPU, where the tiles themselves can have an arbitrary layout.
struct TileSwizzle {
  struct Dim {
    // Describes what varies across this dimension.
    enum class Kind : int8_t {
      // This dimension is internal to one intrinsic on one thread. This
      // is only seen for intrinsic operands that are themselves vectors.
      // For example, with AMD MFMA, for the MFMA_F32_16x16x4_F32 intrinsic,
      // the C-matrix operand is a vector of 4 floats already at the level of
      // one intrinsic on one thread. That dimension of size 4 is 'Internal'.
      Internal,
      // This dimension is internal to one intrinsic, but is across threads.
      // For example, with AMD MFMA, for the MFMA_F32_16x16x4_F32 intrinsic,
      // the A-matrix tile has shape 16x4, and these two dimensions of size 16
      // and 4 are 'CrossThread': neither is visible at the single-thread level
      // (in the intrinsic itself, the A-matrix operand is a single scalar) but
      // as we move along these dimensions, we are moving over the 64 threads
      // of the subgroup.
      //
      // Another example of cross-thread dimensions is in kernels that are
      // "unrolled" across subgroups. Such dimensions are cross-subgroup, so in
      // particular they are cross-thread.
      CrossThread,
      // This dimensions is across intrinsics, as in, actual instructions in the
      // generated code. In other words, it is an actual unrolling factor,
      // resulting in this many more instructions being generated and executed
      // on each thread/subgroup.
      CrossIntrinsic
    };

    Kind kind = Kind::Internal;

    // The size of the dimension.
    int16_t size = 0;

    // Support constructing from any size type.
    template <typename T>
    Dim(Kind kind, T size) : kind(kind), size(size) {}
  };

  using ExpandShapeDimVectorType = llvm::SmallVector<Dim, 4>;
  using ExpandShapeType = llvm::SmallVector<ExpandShapeDimVectorType>;

  // This vector-of-vectors contains all the information needed to generate
  // a `tensor.expand_shape` creating additional internal dimensions into the
  // tile. For example, expandShape = [[16], [4, 2]] means that the original
  // tile shape [16, 8] gets expanded such that the first dimension 16 is left
  // unchanged, and the second dimension 8 gets split into two internal dims
  // of size 4 and 2.
  ExpandShapeType expandShape;
  // This permutation vector applies to the expanded dimensions and is used
  // to generate a `linalg.transpose` changing the layout of the tile. For
  // example, permutation[0] dictates which of the expanded dimensions becomes
  // the leading dimension of the layout.
  llvm::SmallVector<int64_t> permutation;
};

using ScalableTileFlags = SmallVector<bool>;
/// Container of information needed to materialize the layout transformations.
struct MaterializeEncodingInfo {
  // The next 3 fields are used to create a `linalg.pack` or `linalg.unpack` op,
  // changing the overall layout between row-major and tiled (where each tile is
  // row-major).
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;

  // The optional swizzle, see the comment on TileSwizzle. Only used on GPU.
  std::optional<TileSwizzle> swizzle;
  // The optional scalable tiles array
  std::optional<ScalableTileFlags> scalableTiles;
};

} // namespace mlir::iree_compiler::IREE::Codegen
#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IR_IREECODEGENTYPES_H_
