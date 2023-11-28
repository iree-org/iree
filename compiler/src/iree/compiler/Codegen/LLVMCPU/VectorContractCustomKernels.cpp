// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Returns true if `contractionOp` is of the form
//   matrix * transposed_matrix.
// That is, if there are 2 parallel iterators, say M and N, 1 additive reduction
// iterator, say K, and the indexing maps are {{M, K}, {N, K}, {M, N}}.
static bool isMatrixTimesMatrixTransposed(vector::ContractionOp contractionOp) {
  // Check that the reduction is additive.
  if (contractionOp.getKind() != vector::CombiningKind::ADD) {
    return false;
  }
  // Check that there are 2 parallel and 1 reduction iterators.
  auto iteratorTypes = contractionOp.getIteratorTypes().getValue();
  if (iteratorTypes.size() != 3) {
    return false;
  }
  SmallVector<int, 3> parallelIterators;
  SmallVector<int, 3> reductionIterators;
  for (int i = 0; i < 3; i++) {
    if (vector::isParallelIterator(iteratorTypes[i])) {
      parallelIterators.push_back(i);
    } else if (vector::isReductionIterator(iteratorTypes[i])) {
      reductionIterators.push_back(i);
    } else {
      return false;
    }
  }
  if (parallelIterators.size() != 2 || reductionIterators.size() != 1) {
    return false;
  }
  // Give the found iterators some idiomatic names.
  const int MIter = parallelIterators[0];
  const int NIter = parallelIterators[1];
  const int KIter = reductionIterators[0];
  // Check that there are 3 indexing maps.
  auto indexingMaps = contractionOp.getIndexingMapsArray();
  if (indexingMaps.size() != 3) {
    return false;
  }
  // Check that the indexing maps have the expected form.
  const int expectedMapResults[3][2] = {
      {MIter, KIter}, {NIter, KIter}, {MIter, NIter}};
  for (int m = 0; m < 3; ++m) {
    auto map = indexingMaps[m];
    if (map.getNumDims() != 3 || map.getNumResults() != 2) {
      return false;
    }
    for (int r = 0; r < 2; ++r) {
      int actualMapResult =
          cast<AffineDimExpr>(map.getResults()[r]).getPosition();
      if (actualMapResult != expectedMapResults[m][r]) {
        return false;
      }
    }
  }
  return true;
}

// Returns true if `contractionOp` is of the form
//   matrix * transposed_matrix
// where matrix is a vector<{mSize}x{kSize}xType>, and
// transposed_matrix is a vector<{nSize}x{kSize}xType>.
//
// Also returns true if the above condition is met after swapping
// mSize<->nSize and one of these two values is 1, and `transpose` is not null.
// In that case, the output-param `*transpose` is set to true. Rationale: we
// want to use the same kernel for vector*matrix and matrix*vector. The good
// thing with MMT, namely
//
//    A * Transpose(B)
//
// is that swapping A and B merely transposes the result:
//
//    B * Transpose(A) = Transpose( A * Transpose(B) )
//
// This opens the possibility of reducing vector*matrix to matrix*vector
// by merely swappign LHS<->RHS. Why is this specific to the case where one of
// the sides is a vector? Because transposing the result is not OK in general,
// we don't want to write out the result accumulators in the wrong storage
// order. However, when one of the two sides is a vector, so is the result
// accumulator, and for a vector shape (i.e. Mx1 or 1xN), storage orders do not
// matter.
static bool matchMMT(vector::ContractionOp contractionOp, int64_t mSize,
                     int64_t kSize, int64_t nSize, bool *transpose = nullptr) {
  if (!isMatrixTimesMatrixTransposed(contractionOp)) {
    return false;
  }
  VectorType lhsType = llvm::cast<VectorType>(contractionOp.getLhs().getType());
  VectorType rhsType = llvm::cast<VectorType>(contractionOp.getRhs().getType());
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  if (lhsShape[1] != kSize || rhsShape[1] != kSize) {
    return false;
  }
  if (lhsShape[0] == mSize && rhsShape[0] == nSize) {
    return true;
  }
  if (lhsShape[0] == nSize && rhsShape[0] == mSize && transpose != nullptr) {
    *transpose = true;
    return true;
  }
  return false;
}

// `promotedResult` is required to be a Vector.
// If its VectorType does not have `promotedType` as its element type, or
// the operand to the type-promotion op is not `unpromotedType` returns a null
// Value.
// If `unpromotedType == promotedType`, return `promotedResult` unchanged.
// Otherwise, checks that `promotedResult` is defined by a type-promotion op
// (such as arith::ExtSIOp) promoting from `unpromotedType` to `promotedType`,
// and returns the input of that promotion op.
// Note that this only looks at the immediately defining operation, so we likely
// want to have earlier passes that sink widening operations as far down as
// possible, which is probably just good regardless.
static Value getUnpromotedInput(Type unpromotedType, Type promotedType,
                                Value promotedResult) {
  VectorType promotedResultVectorType =
      llvm::cast<VectorType>(promotedResult.getType());
  if (promotedResultVectorType.getElementType() != promotedType) {
    return nullptr;
  }
  if (unpromotedType == promotedType) {
    return promotedResult;
  }
  // TODO: handle promotion of floating point types. Not doing it for now as
  // it wouldn't be exercised.
  auto extSIOp = promotedResult.getDefiningOp<arith::ExtSIOp>();
  if (!extSIOp) {
    return nullptr;
  }
  Value extInput = extSIOp.getIn();
  if (llvm::cast<VectorType>(extInput.getType()).getElementType() !=
      unpromotedType) {
    return nullptr;
  }
  return extInput;
}

// Helper to create a 1D, contiguous slice of a 1D vector.
static Value extract1DSlice(PatternRewriter &rewriter, Location loc,
                            VectorType dstVecType, Value input, int position) {
  assert(input.getType().cast<VectorType>().getRank() == 1);
  assert(dstVecType.getRank() == 1);
  std::array<int64_t, 1> offsets{position};
  std::array<int64_t, 1> strides{1};
  return rewriter.create<vector::ExtractStridedSliceOp>(
      loc, input, offsets, dstVecType.getShape(), strides);
}

// Helper to extract an element of a 1D vector.
static Value extract(PatternRewriter &rewriter, Location loc, Value input,
                     int position) {
  VectorType vectorType = llvm::cast<VectorType>(input.getType());
  assert(vectorType.getRank() == 1);
  (void)vectorType;
  std::array<int64_t, 1> offsets{position};
  return rewriter.create<vector::ExtractOp>(loc, input, offsets);
}

// Helper to flatten a N-dimensional vector to a 1D vector.
static Value flatten(PatternRewriter &rewriter, Location loc, Value vector) {
  VectorType inputVecType = llvm::cast<VectorType>(vector.getType());
  VectorType dstType = VectorType::get(inputVecType.getNumElements(),
                                       inputVecType.getElementType());
  return rewriter.create<vector::ShapeCastOp>(loc, dstType, vector);
}

// Describes a kernel. This struct is kept small to separate the kernels
// themselves from the MLIR-specific generators consuming them
// (see MMTKernelGenerator).
//
// There is some redundancy among this struct's fields: see the relationships
// between fields that are enforced in validate(). This redundancy helps:
// (1) Avoid having to perform divisions (performance concern, and readability
//         concern as would care for these divisions to be exact).
// (2) Be explicit about the size of the vectors involved in the kernel's
//         "calling convention".
struct MMTKernel {
  enum class ScalarType : int8_t { None, I8, I32, F32 };
  // Element type of the LHS vectors.
  ScalarType lhsType = ScalarType::None;
  // Element type of the RHS vectors.
  ScalarType rhsType = ScalarType::None;
  // Element type of the Accumulator and output vectors.
  ScalarType accType = ScalarType::None;
  // Number of rows of the LHS and Accumulator tile.
  int8_t m0 = 0;
  // Reduction dimension, i.e. number of columns of the LHS.
  int8_t k0 = 0;
  // Number of rows of the RHS (note that the operation being targeted, MMT,
  // is matrix multiplication with a *transposed* RHS)
  int8_t n0 = 0;
  // Number of LHS elements in the type of register to be used for the LHS.
  // This is > 1 if SIMD registers are to be used.
  // Note: LHS/RHS/Accumulator may use registers of different sizes.
  int8_t lhsRegSize = 0;
  // Number of RHS elements fitting in the type of register to be used for RHS.
  int8_t rhsRegSize = 0;
  // Number of Accumulator elements fitting  in the type of register to be used
  // for the accumulator.
  int8_t accRegSize = 0;
  // Number of registers needed to hold the LHS.
  int8_t lhsRegs = 0;
  // Number of registers needed to hold the RHS.
  int8_t rhsRegs = 0;
  // Number of registers needed to hold the Accumulator.
  int8_t accRegs = 0;
  // If not null, points to the inline asm code template for this kernel.
  // Register operands for the LHS, RHS and Accumulator are to be referenced as
  // $(lhs:<i>), $(rhs:<i>), $(acc:<i>) respectively, where i is a decimal
  // integer specifying the i-th register for each case (numbered independently,
  // so each starts at 0).
  const char *asmImpl = nullptr;
  // If not null, points to the clobbers list, i.e. the list of registers
  // that the compiler will reserve for this inline asm block's use, in addition
  // to the ones implicitly allocated for the declared inputs and outputs. Using
  // C inline_asm syntax: comma-separated list of raw register names e.g.
  // "v14,v15"
  const char *asmClobbers = nullptr;

  void validate() const {
    assert(m0 * k0 == lhsRegSize * lhsRegs); // number of elements of LHS
    assert(n0 * k0 == rhsRegSize * rhsRegs); // number of elements of RHS
    assert(m0 * n0 == accRegSize * accRegs); // number of elements of Accum
    assert(lhsType != ScalarType::None);
    assert(rhsType != ScalarType::None);
    assert(accType != ScalarType::None);
  }
};

// i8*i8->i32 kernel for Aarch64 NEON.
//
// Historically certain such kernels [1] required int8 inputs not have the
// value -128, which enabled a different kernel design taking advantage
// of the narrow range to accumulate once within int16 accumulators without
// overflow. These kernels were a 1.5x speedup on some late-2010s out-of-order
// cores (ARM Cortex A57/A72/A73, Apple A6--A12, Samsung Exynos M3), but became
// obsolete with the +dotprod feature (ARM Cortex-A76, Apple A13), and never
// were useful on in-order ARM Cortex-A53/A55. So going forward, they are not
// anymore a useful trade-off even in frameworks (such as TensorFlow Lite) that
// are designed to avoid -128 values. There is a large ecosystem cost in
// maintaining that restriction, and it wouldn't make sense to introduce it now
// in new frameworks such as MLIR or IREE, so the present kernel is general,
// supports arbitrary int8 values and does not try to use such optimizations.
//
// This kernel is needed because: at the moment, the codegen has multiple
// issues. It uses inefficient scalar memory access instructions,
// expands int8 values to int32, and performs slow int32*int32 multiplications:
//   118d8: f0 12 c0 39   ldrsb w16, [x23, #4]
//   ...
//   118f4: 1b 0e 04 4e   dup v27.4s, w16
//   ...
//   11900: 32 97 bb 4e   mla v18.4s, v25.4s, v27.4s
//   11904: 57 97 bb 4e   mla v23.4s, v26.4s, v27.4s
//
//
// [1]:
// https://github.com/google/ruy/blob/2d950b3bfa7ebfbe7a97ecb44b1cc4da5ac1d6f0/ruy/kernel_arm64.cc#L93
MMTKernel MMTKernel_8x1x8_i8i8i32_Aarch64_Baseline_InlineAsm() {
  MMTKernel kernel;
  kernel.lhsType = MMTKernel::ScalarType::I8;
  kernel.rhsType = MMTKernel::ScalarType::I8;
  kernel.accType = MMTKernel::ScalarType::I32;
  kernel.m0 = 8; // shape: 8x1x8, outer-product.
  kernel.k0 = 1; // note: we would have enough registers to widen to 12x1x8
  kernel.n0 = 8; // if needed.
  kernel.lhsRegSize = 8; // LHS NEON register type: int8x8
  kernel.rhsRegSize = 8; // RHS NEON register type: int8x8
  kernel.accRegSize = 4; // Accum NEON register type: int32x4
  kernel.lhsRegs = 1;
  kernel.rhsRegs = 1;
  kernel.accRegs = 16; // = 8*8/4 for 8x8 accumulators, 4 per register
  kernel.asmImpl = R"ASM(
      // NEON does not have instructions to multiply int8 values and accumulate
      // into int32. This kernel sign-extends int8 to int16, then uses
      // smlal[2] to multiply-accumulate int16 values into int32 accumulators.
      sxtl v14.8h, $(lhs:0).8b  // v14.8h = sign-extend LHS int8 to int16
      sxtl v15.8h, $(rhs:0).8b  // v15.8h = sign-extend RHS int8 to int16
      smlal $(acc:0).4s, v15.4h, v14.h[0]
      smlal2 $(acc:1).4s, v15.8h, v14.h[0]
      smlal $(acc:2).4s, v15.4h, v14.h[1]
      smlal2 $(acc:3).4s, v15.8h, v14.h[1]
      smlal $(acc:4).4s, v15.4h, v14.h[2]
      smlal2 $(acc:5).4s, v15.8h, v14.h[2]
      smlal $(acc:6).4s, v15.4h, v14.h[3]
      smlal2 $(acc:7).4s, v15.8h, v14.h[3]
      smlal $(acc:8).4s, v15.4h, v14.h[4]
      smlal2 $(acc:9).4s, v15.8h, v14.h[4]
      smlal $(acc:10).4s, v15.4h, v14.h[5]
      smlal2 $(acc:11).4s, v15.8h, v14.h[5]
      smlal $(acc:12).4s, v15.4h, v14.h[6]
      smlal2 $(acc:13).4s, v15.8h, v14.h[6]
      smlal $(acc:14).4s, v15.4h, v14.h[7]
      smlal2 $(acc:15).4s, v15.8h, v14.h[7]
    )ASM";
  kernel.asmClobbers = "v14,v15";
  return kernel;
}

// i8*i8->i32 kernel for Aarch64 NEON, matrix*vector
//
// This kernel is needed because: at the moment, the codegen is generating
// 177 instructions for this kernel (not peeled).
MMTKernel MMTKernel_8x8x1_i8i8i32_Aarch64_Baseline_InlineAsm() {
  MMTKernel kernel;
  kernel.lhsType = MMTKernel::ScalarType::I8;
  kernel.rhsType = MMTKernel::ScalarType::I8;
  kernel.accType = MMTKernel::ScalarType::I32;
  kernel.m0 = 8; // shape: 8x8x1, matrix*vector
  kernel.k0 = 8;
  kernel.n0 = 1;
  kernel.lhsRegSize = 16; // LHS NEON register type: int8x16
  kernel.rhsRegSize = 8;  // RHS NEON register type: int8x8
  kernel.accRegSize = 4;  // Accum NEON register type: int32x4
  kernel.lhsRegs = 4;     // = 8x8/16 for 8x8 LHS elems, 16 per register
  kernel.rhsRegs = 1;
  kernel.accRegs = 2; // = 8/4 for 8 accumulators, 4 per register
  kernel.asmImpl = R"ASM(
    // This kernel multiplies int8 values into temporary int16 values in
    // registers v8--v15, then performs additions. We can't use
    // multiply-accumulate instructions here because of the lack of an
    // instruction multiplying int8 values and accumulating into int32, and
    // we prefer to avoid the overhead of sign-extending the inputs from int8
    // to int16 in this matrix*vector kernel where the largest matrix is the
    // LHS.
    ins v15.d[1], $(rhs:0).d[0]  // copy 1st half of $(rhs:0) to 2nd half of v15
    smull v8.8h, $(lhs:0).8b, $(rhs:0).8b
    smull2 v9.8h, $(lhs:0).16b, v15.16b
    smull v10.8h, $(lhs:1).8b, $(rhs:0).8b
    smull2 v11.8h, $(lhs:1).16b, v15.16b
    smull v12.8h, $(lhs:2).8b, $(rhs:0).8b
    smull2 v13.8h, $(lhs:2).16b, v15.16b
    smull v14.8h, $(lhs:3).8b, $(rhs:0).8b
    smull2 v15.8h, $(lhs:3).16b, v15.16b
    // Now if we were able to codegen not just this MMT in isolation but
    // a whole loop, we would diverge at this point: instead of doing the full
    // additive reduction that the instructions below do, we would do only
    // minimal reductions to temporary int32 accumulators
    // (e.g. sadalp tmp.4s, v8.8h) and we would defer the rest of the work
    // to the end of the loop. This is an example of how "MMT vector.contract"
    // is not a perfect abstraction for "basic block of a MMT inner loop".
    // Anyway...
    //
    // pairwise additions of int16 lanes to int32.
    // So each result int32 is the sum of 2 products.
    saddlp v8.4s, v8.8h
    saddlp v9.4s, v9.8h
    saddlp v10.4s, v10.8h
    saddlp v11.4s, v11.8h
    saddlp v12.4s, v12.8h
    saddlp v13.4s, v13.8h
    saddlp v14.4s, v14.8h
    saddlp v15.4s, v15.8h
    // pairwise additions of int32s, so each result is the sum of 4 products.
    addp v8.4s, v8.4s, v9.4s
    addp v10.4s, v10.4s, v11.4s
    addp v12.4s, v12.4s, v13.4s
    addp v14.4s, v14.4s, v15.4s
    // pairwise additions of int32s, so each result is the sum of 8 products.
    addp v8.4s, v8.4s, v10.4s
    addp v12.4s, v12.4s, v14.4s
    // Add to destination accumulators
    add $(acc:0).4s, $(acc:0).4s, v8.4s
    add $(acc:1).4s, $(acc:1).4s, v12.4s
    )ASM";
  kernel.asmClobbers = "v8,v9,v10,v11,v12,v13,v14,v15";
  return kernel;
}

// i8*i8->i32 kernel for Aarch64 NEON +dotprod
//
// This kernel is needed because: at the moment, codegen doesn't know how to
// make use of dotprod instructions.
MMTKernel MMTKernel_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm() {
  MMTKernel kernel;
  kernel.lhsType = MMTKernel::ScalarType::I8;
  kernel.rhsType = MMTKernel::ScalarType::I8;
  kernel.accType = MMTKernel::ScalarType::I32;
  kernel.m0 = 8; // shape: 8x4x8. We would have enough registers to widen this
  kernel.k0 = 4; // to 12x4x8 if needed.
  kernel.n0 = 8;
  kernel.lhsRegSize = 16; // LHS NEON register type: int8x16
  kernel.rhsRegSize = 16; // RHS NEON register type: int8x16
  kernel.accRegSize = 4;  // Accum NEON register type: int32x4
  kernel.lhsRegs = 2;     // = 8x4/16 for 8x4 LHS elems, 16 per register
  kernel.rhsRegs = 2;     // = 8x4/16 for 8x4 RHS elems, 16 per register
  kernel.accRegs = 16;    // = 8x8/4 for 8x8 Accum elems, 4 per register
  kernel.asmImpl = R"ASM(
      // Note on the operands ordering: RHS before LHS, because we want
      // to multiply a 4x4 tile from RHS against a row-vector from LHS to
      // produce a row-vector of Accumulators, because the accumulator
      // needs to be row-major.
      sdot $(acc:0).4s, $(rhs:0).16b, $(lhs:0).4b[0]
      sdot $(acc:1).4s, $(rhs:1).16b, $(lhs:0).4b[0]
      sdot $(acc:2).4s, $(rhs:0).16b, $(lhs:0).4b[1]
      sdot $(acc:3).4s, $(rhs:1).16b, $(lhs:0).4b[1]
      sdot $(acc:4).4s, $(rhs:0).16b, $(lhs:0).4b[2]
      sdot $(acc:5).4s, $(rhs:1).16b, $(lhs:0).4b[2]
      sdot $(acc:6).4s, $(rhs:0).16b, $(lhs:0).4b[3]
      sdot $(acc:7).4s, $(rhs:1).16b, $(lhs:0).4b[3]
      sdot $(acc:8).4s, $(rhs:0).16b, $(lhs:1).4b[0]
      sdot $(acc:9).4s, $(rhs:1).16b, $(lhs:1).4b[0]
      sdot $(acc:10).4s, $(rhs:0).16b, $(lhs:1).4b[1]
      sdot $(acc:11).4s, $(rhs:1).16b, $(lhs:1).4b[1]
      sdot $(acc:12).4s, $(rhs:0).16b, $(lhs:1).4b[2]
      sdot $(acc:13).4s, $(rhs:1).16b, $(lhs:1).4b[2]
      sdot $(acc:14).4s, $(rhs:0).16b, $(lhs:1).4b[3]
      sdot $(acc:15).4s, $(rhs:1).16b, $(lhs:1).4b[3]
    )ASM";
  return kernel;
}

// i8*i8->i32 kernel for Aarch64 NEON +dotprod, matrix*vector
//
// This kernel is needed because: at the moment, codegen doesn't know how to
// make use of dotprod instructions.
MMTKernel MMTKernel_8x4x1_i8i8i32_Aarch64Dotprod_InlineAsm() {
  MMTKernel kernel;
  kernel.lhsType = MMTKernel::ScalarType::I8;
  kernel.rhsType = MMTKernel::ScalarType::I8;
  kernel.accType = MMTKernel::ScalarType::I32;
  kernel.m0 = 8; // shape: 8x4x1.
  kernel.k0 = 4;
  kernel.n0 = 1;
  kernel.lhsRegSize = 16; // LHS NEON register type: int8x16
  kernel.rhsRegSize = 4;  // RHS NEON register type: int8x4. This is very small
                          // and forces sub-optimal codegen. This needs to be
                          // widened by peeling the surrounding loop, not by
                          // increasing the k0 of this MMT, which would change
                          // the data layout in an unwanted way.
  kernel.accRegSize = 4;  // LHS NEON register type: int8x16
  kernel.lhsRegs = 2;     // = 8x4/16 for 8x4 LHS elems, 16 per register
  kernel.rhsRegs = 1;     // = 4/4 for 4 LHS elems, 4 per register
  kernel.accRegs = 2;     // = 8/4 for 8 Accum elems, 4 per register
  kernel.asmImpl = R"ASM(
      sdot $(acc:0).4s, $(lhs:0).16b, $(rhs:0).4b[0]
      sdot $(acc:1).4s, $(lhs:1).16b, $(rhs:0).4b[0]
    )ASM";
  return kernel;
}

// i8*i8->i32 kernel for Aarch64 NEON +i8mm
//
// This kernel is needed because: at the moment, codegen doesn't know how to
// make use of i8mm instructions.
MMTKernel MMTKernel_8x8x8_i8i8i32_Aarch64I8mm_InlineAsm() {
  MMTKernel kernel;
  kernel.lhsType = MMTKernel::ScalarType::I8;
  kernel.rhsType = MMTKernel::ScalarType::I8;
  kernel.accType = MMTKernel::ScalarType::I32;
  kernel.m0 = 8; // shape: 8x8x8. We would have enough registers to widen this
  kernel.k0 = 8; // to 12x8x8 if needed.
  kernel.n0 = 8;
  kernel.lhsRegSize = 16; // LHS NEON register type: int8x16
  kernel.rhsRegSize = 16; // RHS NEON register type: int8x16
  kernel.accRegSize = 4;  // Accum NEON register type: int32x4
  kernel.lhsRegs = 4;     // = 8x8/16 for 8x4 LHS elems, 16 per register
  kernel.rhsRegs = 4;     // = 8x8/16 for 8x4 RHS elems, 16 per register
  kernel.accRegs = 16;    // = 8x8/4 for 8x8 Accum elems, 4 per register
  kernel.asmImpl = R"ASM(
      // What's with the horrendous shuffles (zip, uzp instructions) ?
      // The smmla instruction works with a 2x2 accumulator tile.
      // So at the moment, given the MMT vector.contract representation of
      // the basic block, we have to perform this re-tiling to 2x2 tiles.
      //
      // This is not really optimized -- just provided to help shape the next
      // stage of the discussion, which will be how we change the abstractions
      // to resolve this.
      //
      // For instance, if we compiled a whole loop at once, we would only need
      // to do so at the start and at the end of the loop. Or even without
      // handing the whole loop to asm, we could make the vector.contract
      // higher-dimensional to allow representing this nested tiled layout.
      // One thing that we should keep in mind though is that this unusual
      // 2x2 tiled layout is specific to matrix multiplication instructions.
      // If the matmul kernel were fused into consumer ops, those would probably
      // prefer not to deal with a 2x2 tiled layout.
      //
      // We also can't easily generalize from this to what will be ARMv9+SME.
      // There, the matmul instruction will also produce a 2D matrix tile,
      // but that will be much wider, 16x16, and itself row-major, so that when
      // store it back to NEON/SVE registers, each of those will be contained
      // within one row. Even if we put multiple such 16x16 tiles side-by-side
      // in the overall kernel, that will still be at a scale larger than
      // individual NEON/SVE registers.
      //
      // Rows 0-1 of the 8x8 accumulator tile
      zip1 v28.2d, $(acc:0).2d, $(acc:2).2d
      zip2 v29.2d, $(acc:0).2d, $(acc:2).2d
      zip1 v30.2d, $(acc:1).2d, $(acc:3).2d
      zip2 v31.2d, $(acc:1).2d, $(acc:3).2d
      smmla v28.4s, $(lhs:0).16b, $(rhs:0).16b
      smmla v29.4s, $(lhs:0).16b, $(rhs:1).16b
      smmla v30.4s, $(lhs:0).16b, $(rhs:2).16b
      smmla v31.4s, $(lhs:0).16b, $(rhs:3).16b
      uzp1 $(acc:0).2d, v28.2d, v29.2d
      uzp1 $(acc:1).2d, v30.2d, v31.2d
      uzp2 $(acc:2).2d, v28.2d, v29.2d
      uzp2 $(acc:3).2d, v30.2d, v31.2d
      // Rows 2-3 of the 8x8 accumulator tile
      zip1 v28.2d, $(acc:4).2d, $(acc:6).2d
      zip2 v29.2d, $(acc:4).2d, $(acc:6).2d
      zip1 v30.2d, $(acc:5).2d, $(acc:7).2d
      zip2 v31.2d, $(acc:5).2d, $(acc:7).2d
      smmla v28.4s, $(lhs:1).16b, $(rhs:0).16b
      smmla v29.4s, $(lhs:1).16b, $(rhs:1).16b
      smmla v30.4s, $(lhs:1).16b, $(rhs:2).16b
      smmla v31.4s, $(lhs:1).16b, $(rhs:3).16b
      uzp1 $(acc:4).2d, v28.2d, v29.2d
      uzp1 $(acc:5).2d, v30.2d, v31.2d
      uzp2 $(acc:6).2d, v28.2d, v29.2d
      uzp2 $(acc:7).2d, v30.2d, v31.2d
      // Rows 4-5 of the 8x8 accumulator tile
      zip1 v28.2d, $(acc:8).2d, $(acc:10).2d
      zip2 v29.2d, $(acc:8).2d, $(acc:10).2d
      zip1 v30.2d, $(acc:9).2d, $(acc:11).2d
      zip2 v31.2d, $(acc:9).2d, $(acc:11).2d
      smmla v28.4s, $(lhs:2).16b, $(rhs:0).16b
      smmla v29.4s, $(lhs:2).16b, $(rhs:1).16b
      smmla v30.4s, $(lhs:2).16b, $(rhs:2).16b
      smmla v31.4s, $(lhs:2).16b, $(rhs:3).16b
      uzp1 $(acc:8).2d, v28.2d, v29.2d
      uzp1 $(acc:9).2d, v30.2d, v31.2d
      uzp2 $(acc:10).2d, v28.2d, v29.2d
      uzp2 $(acc:11).2d, v30.2d, v31.2d
      // Rows 6-7 of the 8x8 accumulator tile
      zip1 v28.2d, $(acc:12).2d, $(acc:14).2d
      zip2 v29.2d, $(acc:12).2d, $(acc:14).2d
      zip1 v30.2d, $(acc:13).2d, $(acc:15).2d
      zip2 v31.2d, $(acc:13).2d, $(acc:15).2d
      smmla v28.4s, $(lhs:3).16b, $(rhs:0).16b
      smmla v29.4s, $(lhs:3).16b, $(rhs:1).16b
      smmla v30.4s, $(lhs:3).16b, $(rhs:2).16b
      smmla v31.4s, $(lhs:3).16b, $(rhs:3).16b
      uzp1 $(acc:12).2d, v28.2d, v29.2d
      uzp1 $(acc:13).2d, v30.2d, v31.2d
      uzp2 $(acc:14).2d, v28.2d, v29.2d
      uzp2 $(acc:15).2d, v30.2d, v31.2d
    )ASM";
  kernel.asmClobbers = "v28,v29,v30,v31";
  return kernel;
}

// TODO:
// i8*i8->i32 kernel for Aarch64 NEON +i8mm, matrix*vector:
// Not implemented yet. Due to the shape of the smmla instruction, such a kernel
// would utilize only 50%. It would still be somewhat faster than the dotprod
// matrix*vector kernel at the moment because reading 64bits into a NEON
// register is faster than reading 32bits twice. That's a shallow advantage that
// might vanish once the vector.contract abstraction layer above kernels is
// improved. Another reason why it's not implemented yet is it would have shape
// 8x8x1, same as the aarch64 baseline matrix*vector i8 kernel, so we would need
// a "kernel benefit" system to cleanly express the preference for the i8mm
// kernel.

// f32*f32->f32 kernel for Aarch64 NEON
//
// Note: this asm kernel isn't needed. The default vector.contract
// lowerings already result in essentially the same code. This is included for
// now for completeness, as we need the f32 matrix*vector kernel below anyway.
MMTKernel MMTKernel_8x1x8_f32f32f32_Aarch64_Baseline_InlineAsm() {
  MMTKernel kernel;
  kernel.lhsType = MMTKernel::ScalarType::F32;
  kernel.rhsType = MMTKernel::ScalarType::F32;
  kernel.accType = MMTKernel::ScalarType::F32;
  kernel.m0 = 8;
  kernel.k0 = 1;
  kernel.n0 = 8;
  kernel.lhsRegSize = 4;
  kernel.rhsRegSize = 4;
  kernel.accRegSize = 4;
  kernel.lhsRegs = 2;
  kernel.rhsRegs = 2;
  kernel.accRegs = 16;
  kernel.asmImpl = R"ASM(
      fmla $(acc:0).4s, $(rhs:0).4s, $(lhs:0).s[0]
      fmla $(acc:1).4s, $(rhs:1).4s, $(lhs:0).s[0]
      fmla $(acc:2).4s, $(rhs:0).4s, $(lhs:0).s[1]
      fmla $(acc:3).4s, $(rhs:1).4s, $(lhs:0).s[1]
      fmla $(acc:4).4s, $(rhs:0).4s, $(lhs:0).s[2]
      fmla $(acc:5).4s, $(rhs:1).4s, $(lhs:0).s[2]
      fmla $(acc:6).4s, $(rhs:0).4s, $(lhs:0).s[3]
      fmla $(acc:7).4s, $(rhs:1).4s, $(lhs:0).s[3]
      fmla $(acc:8).4s, $(rhs:0).4s, $(lhs:1).s[0]
      fmla $(acc:9).4s, $(rhs:1).4s, $(lhs:1).s[0]
      fmla $(acc:10).4s, $(rhs:0).4s, $(lhs:1).s[1]
      fmla $(acc:11).4s, $(rhs:1).4s, $(lhs:1).s[1]
      fmla $(acc:12).4s, $(rhs:0).4s, $(lhs:1).s[2]
      fmla $(acc:13).4s, $(rhs:1).4s, $(lhs:1).s[2]
      fmla $(acc:14).4s, $(rhs:0).4s, $(lhs:1).s[3]
      fmla $(acc:15).4s, $(rhs:1).4s, $(lhs:1).s[3]
    )ASM";
  return kernel;
}

// f32*f32->f32 kernel for Aarch64 NEON, matrix*vector
//
// Note: this is about the most naive possible SIMD kernel here, and it should
// not be needed as this should be an easy case for codegen. Here we are very
// limited in what we can do as a MMT vector.contract lowering - to make a
// better kernel, we would need to peel the surrounding loop, and implement
// a larger vector.contract with an additional parallel iterator, accumulating
// into more separate registers, deferring reduction to the end of the loop.
//
// And yet, this kernel is currently needed, because at the moment this is what
// the codegen generates:
//   10d08: 90 44 c1 ac   ldp     q16, q17, [x4], #32
//   10d0c: c6 04 00 f1   subs    x6, x6, #1
//   10d10: 13 42 10 6e   ext     v19.16b, v16.16b, v16.16b, #8
//   10d14: 34 42 11 6e   ext     v20.16b, v17.16b, v17.16b, #8
//   10d18: b2 44 40 bc   ldr     s18, [x5], #4
//   10d1c: 40 ce 30 0e   fmla    v0.2s, v18.2s, v16.2s
//   10d20: 47 12 b0 0f   fmla    v7.2s, v18.2s, v16.s[1]
//   10d24: 46 1a b0 0f   fmla    v6.2s, v18.2s, v16.s[3]
//   10d28: 42 ce 31 0e   fmla    v2.2s, v18.2s, v17.2s
//   10d2c: 41 12 b1 0f   fmla    v1.2s, v18.2s, v17.s[1]
//   10d30: 45 ce 33 0e   fmla    v5.2s, v18.2s, v19.2s
//   10d34: 44 ce 34 0e   fmla    v4.2s, v18.2s, v20.2s
//   10d38: 43 1a b1 0f   fmla    v3.2s, v18.2s, v17.s[3]
//   10d3c: 61 fe ff 54   b.ne    0x10d08 <.text+0x770>
//
// This is effectively non-SIMD, since each of the 8 fmla here does one useful
// scalar multiplication (note: the ldr s18 instruction loaded one float into
// the first lane of v18.4s and zeroed the other 3 lanes).
MMTKernel MMTKernel_8x1x1_f32f32f32_Aarch64_Baseline_InlineAsm() {
  MMTKernel kernel;
  kernel.lhsType = MMTKernel::ScalarType::F32;
  kernel.rhsType = MMTKernel::ScalarType::F32;
  kernel.accType = MMTKernel::ScalarType::F32;
  kernel.m0 = 8;
  kernel.k0 = 1;
  kernel.n0 = 1;
  kernel.lhsRegSize = 4;
  kernel.rhsRegSize = 1;
  kernel.accRegSize = 4;
  kernel.lhsRegs = 2;
  kernel.rhsRegs = 1;
  kernel.accRegs = 2;
  kernel.asmImpl = R"ASM(
      fmla $(acc:0).4s, $(lhs:0).4s, $(rhs:0).s[0]
      fmla $(acc:1).4s, $(lhs:1).4s, $(rhs:0).s[0]
    )ASM";
  return kernel;
}

// Constructs the mlir::Type corresponding to a scalar type.
Type mlirType(MLIRContext *context, MMTKernel::ScalarType t) {
  switch (t) {
  case MMTKernel::ScalarType::None:
    break;
  case MMTKernel::ScalarType::I8:
    return IntegerType::get(context, 8, IntegerType::Signless);
  case MMTKernel::ScalarType::I32:
    return IntegerType::get(context, 32, IntegerType::Signless);
  case MMTKernel::ScalarType::F32:
    return FloatType::getF32(context);
  }
  assert(false);
  return Type();
}

// This class is a helper for patterns generating custom kernels based on
// MMTKernel structs.
class MMTKernelGenerator {
public:
  MMTKernelGenerator(MLIRContext *context, MMTKernel kernel,
                     IREE::HAL::ExecutableTargetAttr target)
      : context(context), kernel(kernel), target(target) {
    kernel.validate();
  }
  // Generates the kernel. Returns the output accumulator values.
  SmallVector<Value> generate(PatternRewriter &rewriter, Location loc,
                              ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                              ArrayRef<Value> acc) {
    validateOperands(lhs, rhs, acc);
    if (kernel.asmImpl) {
      return generateAsm(rewriter, loc, lhs, rhs, acc);
    }
    // In the future we may have alternate generator paths, e.g. 1D intrinsics
    // or other asm paths with a different interface, e.g. handling also
    // the memory load accesses.
    assert(false && "no implementation provided for kernel");
    return {};
  }
  // Returns the MLIR element type (not vector type) of the LHS
  Type getLhsType() const { return mlirType(context, kernel.lhsType); }
  // Returns the MLIR element type (not vector type) of the RHS
  Type getRhsType() const { return mlirType(context, kernel.rhsType); }
  // Returns the MLIR element type (not vector type) of the Accumulator
  Type getAccType() const { return mlirType(context, kernel.accType); }
  // Returns the VectorType of LHS SIMD register vectors
  VectorType getLhsRegVectorType() const {
    return VectorType::get({kernel.lhsRegSize}, getLhsType());
  }
  // Returns the VectorType of RHS SIMD register vectors
  VectorType getRhsRegVectorType() const {
    return VectorType::get({kernel.rhsRegSize}, getRhsType());
  }
  // Returns the VectorType of Accumulator SIMD register vectors
  VectorType getAccRegVectorType() const {
    return VectorType::get({kernel.accRegSize}, getAccType());
  }

private:
  MLIRContext *const context;
  const MMTKernel kernel;
  const IREE::HAL::ExecutableTargetAttr target;

  // Helper for generate(). Asserts sanity of the vector-of-register-vectors.
  void validateOperands(ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                        ArrayRef<Value> acc) {
    auto validate = [](ArrayRef<Value> vals, int expectedSize,
                       VectorType expectedType) {
      assert(vals.size() == expectedSize);
      for (const auto &val : vals) {
        assert(val.getType().dyn_cast<VectorType>() == expectedType);
        (void)val;
      }
      (void)expectedSize;
      (void)expectedType;
    };
    validate(lhs, kernel.lhsRegs, getLhsRegVectorType());
    validate(rhs, kernel.rhsRegs, getRhsRegVectorType());
    validate(acc, kernel.accRegs, getAccRegVectorType());
  }
  // Helper for generateAsmCodeAndConstraints
  std::string getConstraintCode() const {
    if (isAArch64(target)) {
      return "w";
    }
    assert(false && "what constraint code to use on this arch?");
    return {};
  }
  // Helper class to build the constraints string of an inline_asm op.
  class Constraints {
  private:
    // The LLVM inline asm syntax is documented here:
    // https://llvm.org/docs/LangRef.html#inline-assembler-expressions
    SmallVector<std::string> inputs;
    SmallVector<std::string> outputs;
    SmallVector<std::string> tiedInputs;
    SmallVector<std::string> clobbers;

  public:
    enum class Kind { Input, InputOutput };
    // Add a new constraint.
    void add(Kind kind, const std::string &constraintCode) {
      switch (kind) {
      case Kind::Input:
        inputs.push_back(constraintCode);
        return;
      case Kind::InputOutput:
        // An input represented by a number `i` is a tied input, tied to the
        // i-th output.
        tiedInputs.push_back(llvm::itostr(outputs.size()));
        outputs.push_back(std::string("=") + constraintCode);
        return;
      }
      assert(false);
    }
    void setClobbers(const char *rawClobbersStr) {
      assert(clobbers.empty());
      if (!rawClobbersStr) {
        return;
      }
      for (StringRef c : llvm::split(rawClobbersStr, ',')) {
        clobbers.push_back(("~{" + c + "}").str());
      }
    }
    // Returns the constraints string to be passed to the inline_asm op.
    // llvm::concat does not currently support concatenating const-qualified
    // objects, so we can't currently const-qualify this method.
    std::string toString() {
      return llvm::join(
          llvm::concat<std::string>(outputs, inputs, tiedInputs, clobbers),
          ",");
    }
  };
  // Helper for generateAsm. Performs some pre-processing of the kernel's
  // asmImpl. Refer to the comment on kernel::asmImpl.
  void generateAsmCodeAndConstraints(std::string &code,
                                     std::string &constraintsString) {
    assert(code.empty());
    assert(constraintsString.empty());
    // The LLVM inline asm syntax is documented here:
    // https://llvm.org/docs/LangRef.html#inline-assembler-expressions
    Constraints constraints;
    code = kernel.asmImpl;
    // processedIdx is the index of a register in the processed asm.
    // Example:   $5   =>   processedIdx == 5
    int processedIdx = 0;
    auto processOperands = [&](Constraints::Kind constraintKind,
                               const char *name, int count) {
      const std::string &constraintCode = getConstraintCode();
      // unprocessedIdx is the index of a register in the unprocessed asm.
      // Example:   $(lhs:1)   =>   unprocessedIdx == 1
      for (int unprocessedIdx = 0; unprocessedIdx < count;
           ++unprocessedIdx, ++processedIdx) {
        constraints.add(constraintKind, constraintCode);
        // Perform the code replacement for the operand.
        // Example:   $(lhs:1)   =>   $5
        replaceAllSubstrsInPlace(
            code, llvm::formatv("$({0}:{1})", name, unprocessedIdx),
            llvm::formatv("${0}", processedIdx));
      }
    };
    processOperands(Constraints::Kind::InputOutput, "acc", kernel.accRegs);
    processOperands(Constraints::Kind::Input, "lhs", kernel.lhsRegs);
    processOperands(Constraints::Kind::Input, "rhs", kernel.rhsRegs);
    constraints.setClobbers(kernel.asmClobbers);
    constraintsString = constraints.toString();
  }
  // Helper for generate(). Implements the asm path.
  SmallVector<Value> generateAsm(PatternRewriter &rewriter, Location loc,
                                 ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                                 ArrayRef<Value> acc) {
    SmallVector<Value> inputs;
    // First the input operands. Then the input-output operands, which, as far
    // as input constraints are concerned, are *tied* inputs, i.e. refer to
    // the outputs that we list earlier in the constraints string. This is why
    // us passing the inputs BEFORE the input-outputs here actually matches
    // us listing the inputs AFTER the outputs (but BEFORE the tied-inputs) in
    // the constraints string. Not confusing at all!
    inputs.append(lhs.begin(), lhs.end());
    for (const auto &v : rhs) {
      if (llvm::cast<VectorType>(v.getType()).getNumElements() == 1)
        inputs.push_back(extract(rewriter, loc, v, 0));
      else
        inputs.push_back(v);
    }
    inputs.append(acc.begin(), acc.end());
    // Create the inline asm op.
    SmallVector<Type> outputOperandTypes(
        llvm::map_range(acc, [](Value v) { return v.getType(); }));
    auto returnType =
        LLVM::LLVMStructType::getLiteral(context, outputOperandTypes);
    auto dialectAttr =
        LLVM::AsmDialectAttr::get(context, LLVM::AsmDialect::AD_ATT);
    std::string code;
    std::string constraints;
    generateAsmCodeAndConstraints(code, constraints);
    LLVM::InlineAsmOp asmOp = rewriter.create<LLVM::InlineAsmOp>(
        loc, returnType, inputs, code, constraints,
        /*has_side_effects=*/false, /*is_align_stack=*/false, dialectAttr,
        /*operand_attrs=*/ArrayAttr());
    // Extract result vectors from the asm op.
    SmallVector<Value> resVec;
    for (int i = 0; i < kernel.accRegs; ++i) {
      SmallVector<int64_t, 1> position = {i};
      resVec.push_back(
          rewriter.create<LLVM::ExtractValueOp>(loc, asmOp.getRes(), position));
    }
    return resVec;
  }
};

/// Converts matrix-times-matrix-transposed vector.contracts, and possibly also
/// any type-promotion op (such as arith.extsi) on the input operands, to
/// a custom kernel (at the moment a llvm.inline_asm op) provided by the
/// MMTKernel struct.
///
/// For example, in the case of a i8*i8->i32 kernel, the IR being replaced
/// by the llvm.inline_asm op might look like:
///
///     %lhs_i32 = arith.extsi %lhs_i8 : i8 to i32
///     %rhs_i32 = arith.extsi %rhs_i8 : i8 to i32
///     %result = vector.contract [...]
///                 %lhs_i32 : vector<8x4xi32>,
///                 %rhs_i32 : vector<8x4xi32>,
///                 %acc_i32 : vector<8x8xi32>,
///                 [...]
///
class MMTCustomKernelPattern : public OpRewritePattern<vector::ContractionOp> {
private:
  MMTKernel kernel;

public:
  MMTCustomKernelPattern(MLIRContext *context, MMTKernel kernel)
      : OpRewritePattern<vector::ContractionOp>(context), kernel(kernel) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractionOp,
                                PatternRewriter &rewriter) const override {
    // Check if `contractionOp` matches, and obtain the (un-promoted) input
    // LHS and RHS vectors.
    bool transposeKernel = false;
    if (!matchMMT(contractionOp, kernel.m0, kernel.k0, kernel.n0,
                  &transposeKernel)) {
      return failure();
    }
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(contractionOp);
    MMTKernelGenerator generator(rewriter.getContext(), kernel, target);
    Type lhsElemType = generator.getLhsType();
    Type rhsElemType = generator.getRhsType();
    Type accElemType = generator.getAccType();
    VectorType accType =
        llvm::cast<VectorType>(contractionOp.getAcc().getType());
    if (accType.getElementType() != accElemType) {
      return failure();
    }
    Value unpromotedLhs =
        getUnpromotedInput(lhsElemType, accElemType, contractionOp.getLhs());
    Value unpromotedRhs =
        getUnpromotedInput(rhsElemType, accElemType, contractionOp.getRhs());
    if (!unpromotedLhs || !unpromotedRhs) {
      return failure();
    }
    // Prepare the dense array attribute that will be used as the initializer
    // for the destination accumulator vector, before actual values are inserted
    // into it. We do this early because here we need to validate that the
    // destination scalar type is one that we know how to handle.
    VectorType flatAccVectorType =
        VectorType::get({accType.getNumElements()}, accType.getElementType());
    ;
    TypedAttr resultInitializer;
    if (accElemType.isSignlessInteger()) {
      resultInitializer = DenseIntElementsAttr::get(flatAccVectorType, 0);
    } else if (accElemType.isF32()) {
      resultInitializer = DenseFPElementsAttr::get(flatAccVectorType, 0.f);
    } else {
      return failure();
    }
    // `contractionOp` matches, start rewriting it.
    Location loc = contractionOp.getLoc();
    // Flatten the inputs to 1D vectors.
    Value flatLhs = flatten(rewriter, loc, unpromotedLhs);
    Value flatRhs = flatten(rewriter, loc, unpromotedRhs);
    Value flatAcc = flatten(rewriter, loc, contractionOp.getAcc());
    // Slice into SIMD-register-sized 1D input vectors ready to feed to the
    // target SIMD instructions.
    auto sliceIntoRegVectors = [&](int regsCount, VectorType regVectorType,
                                   Value src) {
      SmallVector<Value> regVectors;
      int regSize = regVectorType.getNumElements();
      for (int i = 0; i < regsCount; ++i) {
        regVectors.push_back(
            extract1DSlice(rewriter, loc, regVectorType, src, i * regSize));
      }
      return regVectors;
    };
    VectorType lhsRegVectorType = generator.getLhsRegVectorType();
    VectorType rhsRegVectorType = generator.getRhsRegVectorType();
    VectorType accRegVectorType = generator.getAccRegVectorType();
    Value flatLhsForKernel = transposeKernel ? flatRhs : flatLhs;
    Value flatRhsForKernel = transposeKernel ? flatLhs : flatRhs;
    SmallVector<Value> lhsRegVectors =
        sliceIntoRegVectors(kernel.lhsRegs, lhsRegVectorType, flatLhsForKernel);
    SmallVector<Value> rhsRegVectors =
        sliceIntoRegVectors(kernel.rhsRegs, rhsRegVectorType, flatRhsForKernel);
    SmallVector<Value> accRegVectors =
        sliceIntoRegVectors(kernel.accRegs, accRegVectorType, flatAcc);
    // Generate the kernel!
    SmallVector<Value> resRegVectors = generator.generate(
        rewriter, loc, lhsRegVectors, rhsRegVectors, accRegVectors);
    // Insert the result vectors of size 4 into the overall result vector of
    // size 64, still 1D.
    Value result = rewriter.create<arith::ConstantOp>(loc, flatAccVectorType,
                                                      resultInitializer);
    int accRegNumElements = accRegVectorType.getNumElements();
    for (int i = 0; i < kernel.accRegs; ++i) {
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, resRegVectors[i], result,
          std::array<int64_t, 1>{accRegNumElements * i},
          std::array<int64_t, 1>{1});
    }
    // Cast the result from 1D to 2D and replace the original vector.contract.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(contractionOp, accType,
                                                     result);
    return success();
  }
};

/// Converts matrix-times-matrix-transposed vector.contracts with
/// lhs and rhs inputs defined by arith.extsi promoting from i8 to i32,
///
///     %lhs_i32 = arith.extsi %lhs_i8 : i8 to i32
///     %rhs_i32 = arith.extsi %rhs_i8 : i8 to i32
///     %result = vector.contract [...]
///                 %lhs_i32 : vector<8x4xi32>,
///                 %rhs_i32 : vector<8x4xi32>,
///                 %acc_i32 : vector<8x8xi32>,
///                 [...]
///
/// To vector ops reading directly from the %lhs_i8 and %rhs_i8 values
/// (bypassing the existing arith.extsi) and passing that to a llvm.inline_asm
/// block implementing the matrix multiplication arithmetic using Aarch64
/// dot-product instructions (sdot).
/// It matches the same patterns as MMT_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm
struct MMT_8x4x8_i8i8i32_Aarch64Dotprod_Intrinsics
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractionOp,
                                PatternRewriter &rewriter) const override {
    if (!matchMMT(contractionOp, 8, 4, 8)) {
      return failure();
    }

    Type I8Type = rewriter.getIntegerType(8);
    Type I32Type = rewriter.getIntegerType(32);

    auto acc = contractionOp.getAcc();
    auto lhs = contractionOp.getLhs();
    auto rhs = contractionOp.getRhs();
    if (llvm::cast<VectorType>(acc.getType()).getElementType() != I32Type) {
      return failure();
    }

    Value inLhs = getUnpromotedInput(I8Type, I32Type, lhs);
    Value inRhs = getUnpromotedInput(I8Type, I32Type, rhs);

    if (!inLhs || !inRhs)
      return failure();

    auto loc = contractionOp.getLoc();

    auto int32x4VType = VectorType::get({4}, I32Type);

    std::array<Value, 16> accChunks;
    {
      int idx = 0;
      for (int row = 0; row < 8; ++row) {
        auto accRow = rewriter.create<vector::ExtractOp>(
            loc, acc, ArrayRef<int64_t>{row});
        for (int col = 0; col < 8; col += 4) {
          auto accChunk = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, accRow, ArrayRef<int64_t>{col}, ArrayRef<int64_t>{4},
              ArrayRef<int64_t>{1});
          assert(accChunk.getType() == int32x4VType);
          accChunks[idx++] = accChunk;
        }
      }
    }

    auto int8x4x4VType = VectorType::get({4, 4}, rewriter.getIntegerType(8));
    auto extract4x4 = [&](Value in, int rowOffset, int colOffset) {
      auto chunk = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, in, ArrayRef<int64_t>{rowOffset, colOffset},
          ArrayRef<int64_t>{4, 4}, ArrayRef<int64_t>{1, 1});
      assert(chunk.getType() == int8x4x4VType);
      return chunk;
    };

    std::array<Value, 2> lhsHalves = {extract4x4(inLhs, 0, 0),
                                      extract4x4(inLhs, 4, 0)};
    std::array<Value, 2> rhsHalves = {extract4x4(inRhs, 0, 0),
                                      extract4x4(inRhs, 4, 0)};

    auto int8Zero4x4 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(int8x4x4VType));
    auto sdot = [&](Value acc, Value a, Value b, int64_t lane) -> Value {
      auto bReplicatedLane = rewriter.create<vector::ShuffleOp>(
          loc, b, int8Zero4x4, ArrayRef<int64_t>{lane, lane, lane, lane});

      return rewriter.create<arm_neon::Sdot2dOp>(loc, int32x4VType, acc, a,
                                                 bReplicatedLane);
    };

    std::array<Value, 16> dstChunks;
    {
      int idx = 0;
      for (Value lhs : lhsHalves) {
        for (int lane = 0; lane < 4; ++lane) {
          for (Value rhs : rhsHalves) {
            dstChunks[idx] = sdot(accChunks[idx], rhs, lhs, lane);
            ++idx;
          }
        }
      }
    }

    // Put the results back in the accumulator
    {
      int idx = 0;
      for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; col += 4) {
          acc = rewriter.create<vector::InsertStridedSliceOp>(
              loc, dstChunks[idx++], acc, ArrayRef<int64_t>{row, col},
              ArrayRef<int64_t>{1});
        }
      }
    }
    rewriter.replaceOp(contractionOp, {acc});
    return success();
  }
};

class VectorContractCustomKernelsPass
    : public VectorContractCustomKernelsBase<VectorContractCustomKernelsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, LLVM::LLVMDialect,
                    arm_neon::ArmNeonDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto funcOp = getOperation();
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    populateVectorContractCustomKernelsPatterns(target, patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  IREE::HAL::ExecutableTargetAttr target;
};

} // namespace

void populateVectorContractCustomKernelsPatterns(
    IREE::HAL::ExecutableTargetAttr target, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  if (isAArch64(target)) {
    // TODO: add a "kernel benefit" system whereby if two kernels are available
    // for the same shape and same data types, the fastest one (ie the one
    // using the most powerful available SIMD instructions) is selected.
    // This is incidentally not needed at the moment because currently no two
    // kernels share the exact same shape and data types.
    patterns.add<MMTCustomKernelPattern>(
        context, MMTKernel_8x1x8_f32f32f32_Aarch64_Baseline_InlineAsm());
    patterns.add<MMTCustomKernelPattern>(
        context, MMTKernel_8x1x1_f32f32f32_Aarch64_Baseline_InlineAsm());
    patterns.add<MMTCustomKernelPattern>(
        context, MMTKernel_8x1x8_i8i8i32_Aarch64_Baseline_InlineAsm());
    patterns.add<MMTCustomKernelPattern>(
        context, MMTKernel_8x8x1_i8i8i32_Aarch64_Baseline_InlineAsm());
    if (hasFeature(target, "+dotprod")) {
      if (preferIntrinsicsOverAsm(target)) {
        patterns.add<MMT_8x4x8_i8i8i32_Aarch64Dotprod_Intrinsics>(context);
      } else {
        patterns.add<MMTCustomKernelPattern>(
            context, MMTKernel_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm());
        patterns.add<MMTCustomKernelPattern>(
            context, MMTKernel_8x4x1_i8i8i32_Aarch64Dotprod_InlineAsm());
      }
    }
    if (hasFeature(target, "+i8mm")) {
      patterns.add<MMTCustomKernelPattern>(
          context, MMTKernel_8x8x8_i8i8i32_Aarch64I8mm_InlineAsm());
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createVectorContractCustomKernelsPass() {
  return std::make_unique<VectorContractCustomKernelsPass>();
}

} // namespace iree_compiler
} // namespace mlir
