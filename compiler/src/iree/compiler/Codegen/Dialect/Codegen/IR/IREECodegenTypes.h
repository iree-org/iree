// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IR_IREECODEGENTYPES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IR_IREECODEGENTYPES_H_

#include <cassert>
#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler::IREE::Codegen {

// TileSwizzle describes the layout of a tile. To first approximation, "a tile"
// means any statically-shaped object, but this is specifically intended for
// things like the tiles of tiled operands of a inner_tiled operation, e.g.
// typically a matrix multiplication kernel, and a specific provision is made to
// allow modelling scalable dimensions on ISAs having scalable vectors (e.g. ARM
// SVE/SME and RISC-V RVV).
//
// The word "swizzle" refers to the sequence of operations transforming a tile
// from a row-major input layout into the desired layout.
//
// The basic observation is that no matter how complex the swizzle, it can
// always be achieved by the same sequence of 2 MLIR operations:
// 1. `tensor.expand_shape` to split dimensions into finer dimensions, which in
//    itself is purely formal and does not change layout. This operation creates
//    additional internal dimensions to the tile.
// 2. `linalg.transpose` on the expanded dimensions to change the layout.
//
// Thus, to first approximation, TileSwizzle simply captures the defining
// attributes of these expand_shape and transpose operations:
// 1. The expandShape_ member captures the reassociation of the expand_shape op.
// 2. The permutation_ member captures the permutation of the transpose op.
//
// What we have described so far is more or less equivalent to CuTe layouts and
// some de-fragmentation could be envisioned in the future. However, additional
// expressiveness is added by having the TileSwizzle::Dim nested type hold a
// little more than merely the integer size of the dimension:
// 1. TileSwizzle::Dim::Kind describes what varies across this dimension.
//    By tracking which expanded dimension is cross-thread or cross-intrinsic,
//    the TileSwizzle is self-contained for purposes of thread-distribution and
//    code-generation in a way that a CuTe layout is not.
// 2. For CrossThread dimensions, the distributionFactor allows broadcasting
//    data to multiple threads.
// 3. For Internal dimensions, the symbolicMultiplier_ allows modelling scalable
//    dimensions on ISAs having scalable vectors (e.g. ARM SVE/SME and RISC-V
//    RVV).
class TileSwizzle {
public:
  class Dim {
  public:
    // Describes what varies across this dimension.
    enum class Kind : int8_t {
      // This dimension is internal to one intrinsic on one thread. This
      // is only seen for intrinsic operands that are themselves vectors.
      // For example, with AMD MFMA, for the MFMA_F32_16x16x4_F32 intrinsic,
      // the C-matrix operand is a vector of 4 floats already at the level of
      // one intrinsic on one thread. That dimension of size 4 is 'Internal'.
      Internal,
      // This dimension is internal to one intrinsic, but is across threads.
      // By definition, this only happens on SIMT architectures (GPUs).
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
      // This dimension is across intrinsics, as in, actual instructions in the
      // generated code. In other words, it is an actual unrolling factor,
      // resulting in this many more instructions being generated and executed
      // on each thread/subgroup.
      CrossIntrinsic
    };

    // Describes a symbolic multiplier on this dimension's size. Most dimensions
    // will use the value One, meaning no multiplier. The other values are
    // available to support scalable vector ISAs such as ARM SVE/SME and RISC-V.
    enum class SymbolicMultiplier : int8_t {
      // The multiplier is the constant value 1. This is the common case, used
      // for all but the scalable dimensions.
      One,
      // The multiplier is the `vscale` parameter of the Arm ISA. By definition,
      // it is vector length divided by 128 bits, e.g. vscale=2 means 256 bits.
      // Note that with SME, the value of vscale depends on the streaming mode.
      // The current semantics is that we just allow that dependenced on the
      // streaming mode to exist. Whenever we get to implementing data-tiling
      // with SME, we will find out if this works in practice or if we need to
      // introduce a separate enum value for each streaming mode.
      ArmVscale,
      // The multiplier is the VLEN parameter in the RISC-V ISA, expressed in
      // multiples of 128 bits. This is just a placeholder for future use. I
      // have no idea if 128 bits is the right granularity for this. Note
      // however that we can only multiply, not divide, so the unit better not
      // be too small. If no one unit satisfies all use cases, we can introduce
      // separate enum values for different units.
      RiscvVlenIn128bitUnits
    };

    //
    // Static factory methods to create Dim objects.
    //
    static Dim
    internal(int64_t size,
             SymbolicMultiplier symbolicMultiplier = SymbolicMultiplier::One) {
      Dim dim(Kind::Internal, size);
      dim.symbolicMultiplier_ = symbolicMultiplier;
      return dim;
    }

    static Dim crossIntrinsic(int64_t size) {
      return Dim(Kind::CrossIntrinsic, size);
    }

    static Dim crossThread(int64_t size, int64_t distributionFactor = 1) {
      Dim dim(Kind::CrossThread, size);
      dim.distributionFactor_ = distributionFactor;
      return dim;
    }

    Kind kind() const { return kind_; }
    int size() const { return size_; }
    int distributionFactor() const {
      assert(kind() == Kind::CrossThread &&
             "distributionFactor() only defined for CrossThread dims");
      return distributionFactor_;
    }
    SymbolicMultiplier symbolicMultiplier() const {
      assert(kind() == Kind::Internal &&
             "symbolicMultiplier() only defined for Internal dims");
      return symbolicMultiplier_;
    }

  private:
    Dim(Kind kind, int64_t size) : kind_(kind), size_(size) {}

    Kind kind_ = Kind::Internal;

    // The following members are in a union because they are mutually exclusive,
    // as they are each specific to a different kind of dimension.
    union {
      // Only for CrossThread dimensions.
      // The distribution multiplier on the size of the dimension, for
      // distribution purposes. This describes the situation where multiple
      // threads see the same data. `distributionFactor_` is the number of
      // threads sharing the same data. In that case, the size of the dimension
      // becomes  `distributionFactor_` times larger for distribution purposes.
      // `distributionFactor_` consecutive positions along the dimension are the
      // same data, seen by `distributionFactor_` threads.
      //
      // Note about the zero-initialization of this field: we need to
      // zero-initialize one union member, it doesn't matter which one and it
      // can't be more than one. The reason why we need to zero-initialize is
      // that we want to be able to perform equality comparison as memcmp.
      int8_t distributionFactor_ = 0;

      // Only for Internal dimensions.
      // The symbolic multiplier on the size of the dimension, for
      // scalable purposes, on ISAs that have scalable vectors.
      SymbolicMultiplier symbolicMultiplier_;
    };

    // The size of the dimension.
    // For CrossThread dimensions, this may get multiplied by
    // distributionFactor.
    int16_t size_ = 0;
  };

  using ExpandShapeDimVectorType = SmallVector<Dim, 4>;
  using ExpandShapeType = SmallVector<ExpandShapeDimVectorType>;

  // Returns the total number of expanded dimensions.
  int64_t getExpandedSize() const;

  // Verifies consistency of the tile swizzle:
  // - The permutation size must match the total number of expanded dimensions.
  // - The permutation indices must be valid (within bounds and unique).
  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError) const;

  ExpandShapeType &expandShape() { return expandShape_; }
  const ExpandShapeType &expandShape() const { return expandShape_; }
  SmallVector<int64_t> &permutation() { return permutation_; }
  const SmallVector<int64_t> &permutation() const { return permutation_; }

private:
  // This vector-of-vectors contains all the information needed to generate
  // a `tensor.expand_shape` creating additional internal dimensions into the
  // tile. For example, expandShape = [[16], [4, 2]] means that the original
  // tile shape [16, 8] gets expanded such that the first dimension 16 is left
  // unchanged, and the second dimension 8 gets split into two internal dims
  // of size 4 and 2.
  ExpandShapeType expandShape_;
  // This permutation vector applies to the expanded dimensions and is used
  // to generate a `linalg.transpose` changing the layout of the tile. For
  // example, permutation[0] dictates which of the expanded dimensions becomes
  // the leading dimension of the layout.
  SmallVector<int64_t> permutation_;
};

static_assert(sizeof(TileSwizzle::Dim) == 4,
              "TileSwizzle::Dim should be 4 bytes");

/// Returns the swizzled tile shape, but with dim sizes overwritten with 1 if
/// `predicate` returns false.
SmallVector<int64_t>
sliceSwizzledShape(const TileSwizzle &swizzle,
                   llvm::function_ref<bool(TileSwizzle::Dim)> predicate);

/// Pushes `dim` to the front of `swizzle.expandShape[srcIdx]`, and updates
/// `swizzle.permutation` accordingly.
void expand(TileSwizzle &swizzle, size_t srcIdx, TileSwizzle::Dim dim);

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
