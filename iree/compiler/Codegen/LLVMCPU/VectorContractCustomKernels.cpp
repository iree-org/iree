// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Utils/CustomKernelsTargetInfo.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
  if (contractionOp.kind() != vector::CombiningKind::ADD) {
    return false;
  }
  // Check that there are 2 parallel and 1 reduction iterators.
  auto iteratorTypes = contractionOp.iterator_types().getValue();
  if (iteratorTypes.size() != 3) {
    return false;
  }
  SmallVector<int, 3> parallel_iterators;
  SmallVector<int, 3> reduction_iterators;
  for (int i = 0; i < 3; i++) {
    if (isParallelIterator(iteratorTypes[i])) {
      parallel_iterators.push_back(i);
    } else if (isReductionIterator(iteratorTypes[i])) {
      reduction_iterators.push_back(i);
    } else {
      return false;
    }
  }
  if (parallel_iterators.size() != 2 || reduction_iterators.size() != 1) {
    return false;
  }
  // Give the found iterators some idiomatic names.
  const int MIter = parallel_iterators[0];
  const int NIter = parallel_iterators[1];
  const int KIter = reduction_iterators[0];
  // Check that there are 3 indexing maps.
  auto indexingMaps = contractionOp.indexing_maps().getValue();
  if (indexingMaps.size() != 3) {
    return false;
  }
  // Check that the indexing maps have the expected form.
  const int expectedMapResults[3][2] = {
      {MIter, KIter}, {NIter, KIter}, {MIter, NIter}};
  for (int m = 0; m < 3; ++m) {
    auto map = indexingMaps[m].cast<AffineMapAttr>().getValue();
    if (map.getNumDims() != 3 || map.getNumResults() != 2) {
      return false;
    }
    for (int r = 0; r < 2; ++r) {
      int actualMapResult =
          map.getResults()[r].cast<AffineDimExpr>().getPosition();
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
// transposed_matrix is a vector<{nSize}x{kSize}xType>
static bool isMatrixTimesMatrixTransposedOfGivenShape(
    vector::ContractionOp contractionOp, int64_t mSize, int64_t kSize,
    int64_t nSize) {
  if (!isMatrixTimesMatrixTransposed(contractionOp)) {
    return false;
  }
  VectorType lhsType = contractionOp.lhs().getType().cast<VectorType>();
  VectorType rhsType = contractionOp.rhs().getType().cast<VectorType>();
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  if (lhsShape[0] != mSize || lhsShape[1] != kSize || rhsShape[0] != nSize ||
      rhsShape[1] != kSize) {
    return false;
  }
  return true;
}

// Checks that the Value `extResult` is defined by an arith::ExtSIOp promoting
// from `extSrcType` to `extDstType`, and returns the input of the ExtSIOp.
// Note that this only looks at the immediately defining operation, so we likely
// want to have earlier passes that sink widening operations as far down as
// possible, which is probably just good regardless.
static Value getUnpromotedInput(Type extSrcType, Type extDstType,
                                Value extResult) {
  auto extSIOp = extResult.getDefiningOp<arith::ExtSIOp>();
  if (!extSIOp) {
    return nullptr;
  }
  Value extInput = extSIOp.getIn();
  if (extInput.getType().cast<VectorType>().getElementType() != extSrcType) {
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

// Helper to flatten a N-dimensional vector to a 1D vector.
static Value flatten(PatternRewriter &rewriter, Location loc, Value vector) {
  VectorType inputVecType = vector.getType().cast<VectorType>();
  VectorType dstType = VectorType::get(inputVecType.getNumElements(),
                                       inputVecType.getElementType());
  return rewriter.create<vector::ShapeCastOp>(loc, dstType, vector);
}

// Asserts that i is a power of two, and returns its log2.
// Note: the llvm helpers used internally operate on uint32, but we keep that
// an internal detail as the surrounding code here is all operating on signed
// integers and mixing signed and unsigned would be error-prone.
int8_t exactLog2(int32_t i) {
  assert(i > 0);
  uint32_t u = i;
  assert(llvm::isPowerOf2_32(u));
  return llvm::countTrailingZeros(u);
}

// Helper to handle powers of two size computations without the overhead
// of runtime divisions. Divisions remain very expensive compared to most other
// instructions.  Divisions are of course cheap when the divisor is
// a constant, but a typical use case for us is
//
//    lhsBitWidth / kernel.registerBitWidth
//
// kernel.registerBitWidth is *initialized* from a literal value (say 128) but
// it would be cumbersome to have to preserve its constant-expression status
// throughout.
class PowerOfTwo {
 private:
  int8_t exponent = 0;

 public:
  PowerOfTwo() {}
  explicit PowerOfTwo(int32_t i) : exponent(exactLog2(i)) {}
  int getExponent() const { return exponent; }
  int val() const {
    assert(exponent < 8 * sizeof(int) - 1);
    return 1 << exponent;
  }
};

// Returns i/p, asserting that p divides i. Requires i >= 0.
// Fast: bit shift, not actual div.
int32_t fastExactDiv(int32_t i, PowerOfTwo p) {
  assert(i >= 0 && "exact log of negative number");
  int32_t result = i >> p.getExponent();
  assert(result << p.getExponent() == i && "exact log of non-power of two");
  return result;
}

int32_t operator*(int32_t i, PowerOfTwo p) {
  assert(i >= 0 && "only nonnegative values are supported");
  uint32_t u = i;
  assert(llvm::countLeadingZeros(u) > static_cast<unsigned>(p.getExponent()));
  (void)u;
  return i << p.getExponent();
}

// Describes a kernel. This struct is kept small to separate the kernels
// themselves from the MLIR-specific generators consuming them
// (see MMTKernelGenerator).
struct MMTKernel {
  enum class ScalarType : int8_t { None, I8, I32, F32 };
  // Target architecture. Needed to generate inline asm constraints.
  CustomKernelTargetArch arch = CustomKernelTargetArch::None;
  // Bit width of the Simd registers used by the kernel. Needed to determine
  // how to slice Vectors into register-sized Vectors. Not in general fully
  // determined by the arch as it's typical for each arch to have different
  // collections of SIMD instructions with different widths.
  PowerOfTwo registerBitWidth;
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
  // If not null, points to the inline asm code template for this kernel.
  // Register operands for the LHS, RHS and Accumulator are to be referenced as
  // $(lhs:<i>), $(rhs:<i>), $(acc:<i>) respectively, where i is a decimal
  // integer specifying the i-th register for each case (numbered independently,
  // so each starts at 0).
  const char *implAsm = nullptr;
};

// It's not the end of the world to grow this, but let's be mindful as we have
// so far made the choice to pass MMTKernels by value.
static_assert(sizeof(MMTKernel) == 8 + sizeof(void *), "");

// i8*i8->i32 kernel for Aarch64 NEON +dotprod
MMTKernel MMTKernel_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm() {
  MMTKernel kernel;
  kernel.arch = CustomKernelTargetArch::Aarch64;
  kernel.m0 = 8;
  kernel.k0 = 4;
  kernel.n0 = 8;
  kernel.lhsType = MMTKernel::ScalarType::I8;
  kernel.rhsType = MMTKernel::ScalarType::I8;
  kernel.accType = MMTKernel::ScalarType::I32;
  kernel.registerBitWidth = PowerOfTwo(128);
  kernel.implAsm = R"ASM(
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

// Returns the bit-width ( = 8 * sizeof ) of the given scalar type.
PowerOfTwo bitWidth(MMTKernel::ScalarType t) {
  switch (t) {
    case MMTKernel::ScalarType::None:
      break;
    case MMTKernel::ScalarType::I8:
      return PowerOfTwo(8);
    case MMTKernel::ScalarType::I32:
      return PowerOfTwo(32);
    case MMTKernel::ScalarType::F32:
      return PowerOfTwo(32);
  }
  assert(false);
  return PowerOfTwo();
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
  MMTKernelGenerator(MLIRContext *context, MMTKernel kernel)
      : context(context), kernel(kernel) {}
  // Generates the kernel. Returns the output accumulator values.
  SmallVector<Value> generate(PatternRewriter &rewriter, Location loc,
                              ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                              ArrayRef<Value> acc) {
    validateOperands(lhs, rhs, acc);
    if (kernel.implAsm) {
      return generateAsm(rewriter, loc, lhs, rhs, acc);
    }
    // In the future we may have alternate generator paths, e.g. 1D intrinsics
    // or other asm paths with a different interface, e.g. handling also
    // the memory load accesses.
    assert(false && "no implementation provided for kernel");
    return {};
  }
  // Returns the number of SIMD registers needed for the LHS
  int getLhsRegsCount() const {
    int lhsBitWidth = kernel.m0 * kernel.k0 * bitWidth(kernel.lhsType);
    return fastExactDiv(lhsBitWidth, kernel.registerBitWidth);
  }
  // Returns the number of SIMD registers needed for the RHS
  int getRhsRegsCount() const {
    int rhsBitWidth = kernel.n0 * kernel.k0 * bitWidth(kernel.rhsType);
    return fastExactDiv(rhsBitWidth, kernel.registerBitWidth);
  }
  // Returns the number of SIMD registers needed for the Accumulator
  int getAccRegsCount() const {
    int accBitWidth = kernel.m0 * kernel.n0 * bitWidth(kernel.accType);
    return fastExactDiv(accBitWidth, kernel.registerBitWidth);
  }
  // Returns the MLIR element type (not vector type) of the LHS
  Type getLhsType() const { return mlirType(context, kernel.lhsType); }
  // Returns the MLIR element type (not vector type) of the RHS
  Type getRhsType() const { return mlirType(context, kernel.rhsType); }
  // Returns the MLIR element type (not vector type) of the Accumulator
  Type getAccType() const { return mlirType(context, kernel.accType); }
  // Returns the VectorType of LHS SIMD register vectors
  VectorType getLhsRegVectorType() const {
    return VectorType::get(
        {fastExactDiv(kernel.registerBitWidth.val(), bitWidth(kernel.lhsType))},
        getLhsType());
  }
  // Returns the VectorType of RHS SIMD register vectors
  VectorType getRhsRegVectorType() const {
    return VectorType::get(
        {fastExactDiv(kernel.registerBitWidth.val(), bitWidth(kernel.rhsType))},
        getRhsType());
  }
  // Returns the VectorType of Accumulator SIMD register vectors
  VectorType getAccRegVectorType() const {
    return VectorType::get(
        {fastExactDiv(kernel.registerBitWidth.val(), bitWidth(kernel.accType))},
        getAccType());
  }

 private:
  MLIRContext *context;
  MMTKernel kernel;

  // Helper for generate(). Asserts sanity of the vector-of-register-vectors.
  void validateOperands(ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                        ArrayRef<Value> acc) {
    auto validate = [](ArrayRef<Value> vals, int expectedSize,
                       VectorType expectedElemType) {
      assert(vals.size() == expectedSize);
      for (const auto &val : vals) {
        assert(val.getType().dyn_cast<VectorType>() == expectedElemType);
        (void)val;
      }
      (void)expectedSize;
      (void)expectedElemType;
    };
    validate(lhs, getLhsRegsCount(), getLhsRegVectorType());
    validate(rhs, getRhsRegsCount(), getRhsRegVectorType());
    validate(acc, getAccRegsCount(), getAccRegVectorType());
  }
  // Helper for generateAsmCodeAndConstraints
  std::string getInlineAsmConstraintForSimdRegister() const {
    switch (kernel.arch) {
      case CustomKernelTargetArch::Aarch64:
        return "w";
      case CustomKernelTargetArch::None:
        break;
    }
    assert(false && "Unhandled CustomKernelTargetFeature value");
    return {};
  }
  // Helper for generateAsm. Performs some pre-processing of the kernel's
  // implAsm. Refer to the comment on kernel::implAsm.
  void generateAsmCodeAndConstraints(std::string &code,
                                     std::string &constraints) {
    assert(code.empty());
    assert(constraints.empty());
    // The LLVM inline asm syntax is documented here:
    // https://llvm.org/docs/LangRef.html#inline-assembler-expressions
    std::vector<std::string> outputConstraints;
    std::vector<std::string> inputConstraints;
    std::vector<std::string> tiedInputConstraints;
    code = kernel.implAsm;
    int numberedOperand = 0;
    enum class OperandKind { Input, InputOutput };
    std::string simdRegConstraint = getInlineAsmConstraintForSimdRegister();
    auto processOperands = [&](OperandKind kind, int count, const char *name) {
      for (int i = 0; i < count; ++i) {
        std::string numberedOperandStr = llvm::itostr(numberedOperand++);
        std::string match = llvm::formatv("$({0}:{1})", name, i);
        std::string substitute = std::string("$") + numberedOperandStr;
        replaceAllSubstrsInPlace(code, match, substitute);
        if (kind == OperandKind::InputOutput) {
          outputConstraints.push_back(std::string("=") + simdRegConstraint);
          tiedInputConstraints.push_back(numberedOperandStr);
        } else {
          inputConstraints.push_back(simdRegConstraint);
        }
      }
    };
    processOperands(OperandKind::InputOutput, getAccRegsCount(), "acc");
    processOperands(OperandKind::Input, getLhsRegsCount(), "lhs");
    processOperands(OperandKind::Input, getRhsRegsCount(), "rhs");
    constraints = llvm::join(outputConstraints, ",") + "," +
                  llvm::join(inputConstraints, ",") + "," +
                  llvm::join(tiedInputConstraints, ",");
  }
  // Helper for generate(). Implements the asm path.
  SmallVector<Value> generateAsm(PatternRewriter &rewriter, Location loc,
                                 ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                                 ArrayRef<Value> acc) {
    SmallVector<Value> inputs;
    // First the input operands. This matches how in the constraints we are
    // placing the inputConstraints before the tiedInputConstraints, the latter
    // being the input-output operands.
    inputs.append(lhs.begin(), lhs.end());
    inputs.append(rhs.begin(), rhs.end());
    // Then the input-output operands.
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
    for (int i = 0; i < 16; ++i) {
      resVec.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, getAccRegVectorType(), asmOp.getRes(),
          rewriter.getI64ArrayAttr({i})));
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
    if (!isMatrixTimesMatrixTransposedOfGivenShape(contractionOp, kernel.m0,
                                                   kernel.k0, kernel.n0)) {
      return failure();
    }
    MMTKernelGenerator generator(rewriter.getContext(), kernel);
    Type lhsElemType = generator.getLhsType();
    Type rhsElemType = generator.getRhsType();
    Type accElemType = generator.getAccType();
    VectorType accType = contractionOp.acc().getType().cast<VectorType>();
    if (accType.getElementType() != accElemType) {
      return failure();
    }
    Value unpromotedLhs =
        getUnpromotedInput(lhsElemType, accElemType, contractionOp.lhs());
    Value unpromotedRhs =
        getUnpromotedInput(rhsElemType, accElemType, contractionOp.rhs());
    if (!unpromotedLhs || !unpromotedRhs) {
      return failure();
    }
    // `contractionOp` matches, start rewriting it.
    Location loc = contractionOp.getLoc();
    // Flatten the inputs to 1D vectors.
    Value flatLhs = flatten(rewriter, loc, unpromotedLhs);
    Value flatRhs = flatten(rewriter, loc, unpromotedRhs);
    Value flatAcc = flatten(rewriter, loc, contractionOp.acc());
    // Slice into SIMD-register-sized 1D input vectors ready to feed to the
    // target SIMD instructions.
    auto sliceIntoRegVectors = [&](int size, VectorType regVectorType,
                                   Value src) {
      SmallVector<Value> regVectors;
      int regSize = regVectorType.getNumElements();
      for (int position = 0; position < size; position += regSize) {
        regVectors.push_back(
            extract1DSlice(rewriter, loc, regVectorType, src, position));
      }
      return regVectors;
    };
    VectorType lhsRegVectorType = generator.getLhsRegVectorType();
    VectorType rhsRegVectorType = generator.getRhsRegVectorType();
    VectorType accRegVectorType = generator.getAccRegVectorType();
    const SmallVector<Value> &lhsRegVectors =
        sliceIntoRegVectors(kernel.m0 * kernel.k0, lhsRegVectorType, flatLhs);
    const SmallVector<Value> &rhsRegVectors =
        sliceIntoRegVectors(kernel.n0 * kernel.k0, rhsRegVectorType, flatRhs);
    const SmallVector<Value> &accRegVectors =
        sliceIntoRegVectors(kernel.m0 * kernel.n0, accRegVectorType, flatAcc);
    SmallVector<Value> resRegVectors = generator.generate(
        rewriter, loc, lhsRegVectors, rhsRegVectors, accRegVectors);
    // Insert the result vectors of size 4 into the overall result vector of
    // size 64, still 1D.
    VectorType flatAccVectorType = flatAcc.getType().cast<VectorType>();
    Value result = rewriter.create<arith::ConstantOp>(
        loc, flatAccVectorType,
        DenseIntElementsAttr::get(flatAccVectorType, 0));
    int accRegsCount = generator.getAccRegsCount();
    int accRegNumElements = accRegVectorType.getNumElements();
    for (int i = 0; i < accRegsCount; ++i) {
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
    if (!isMatrixTimesMatrixTransposedOfGivenShape(contractionOp, 8, 4, 8)) {
      return failure();
    }

    Type I8Type = rewriter.getIntegerType(8);
    Type I32Type = rewriter.getIntegerType(32);

    auto acc = contractionOp.acc();
    auto lhs = contractionOp.lhs();
    auto rhs = contractionOp.rhs();
    if (acc.getType().cast<VectorType>().getElementType() != I32Type) {
      return failure();
    }

    Value inLhs = getUnpromotedInput(I8Type, I32Type, lhs);
    Value inRhs = getUnpromotedInput(I8Type, I32Type, rhs);

    if (!inLhs || !inRhs) return failure();

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
    registry.insert<vector::VectorDialect, LLVM::LLVMDialect>();
    if (target_info.has(CustomKernelTargetFeature::Intrinsics)) {
      registry.insert<arm_neon::ArmNeonDialect>();
    }
  }
  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    if (failed(ParseCustomKernelsTargetInfo(arch, features, target_info))) {
      return failure();
    }
    if (intrinsics) {
      target_info.add(CustomKernelTargetFeature::Intrinsics);
    }
    return success();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateVectorContractCustomKernelsPatterns(target_info, patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  CustomKernelsTargetInfo target_info;
};

}  // namespace

void populateVectorContractCustomKernelsPatterns(
    const CustomKernelsTargetInfo &target_info, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  if (target_info.has(CustomKernelTargetFeature::Aarch64Dotprod)) {
    if (target_info.has(CustomKernelTargetFeature::Intrinsics)) {
      patterns.add<MMT_8x4x8_i8i8i32_Aarch64Dotprod_Intrinsics>(context);
    } else {
      patterns.add<MMTCustomKernelPattern>(
          context, MMTKernel_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm());
    }
  }
}

std::unique_ptr<OperationPass<FuncOp>> createVectorContractCustomKernelsPass() {
  return std::make_unique<VectorContractCustomKernelsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
