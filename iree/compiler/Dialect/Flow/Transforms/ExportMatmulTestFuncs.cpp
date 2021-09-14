// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <random>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Modules/Check/IR/CheckDialect.h"
#include "iree/compiler/Dialect/Modules/Check/IR/CheckOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Functions that we expect to find in the input MLIR file.
// They are helpers, not test code per se --- all of the
// test code is going to be generated here, except for these
// helper functions that we'll be able to just call.
struct StandardFuncOps {
  // Reference matmul implementation that we'll compare against.
  FuncOp referenceMatmul;
  // Generate a zero matrix. Used to generate LHS and RHS test input matrices.
  FuncOp zeroMatrix;
  // Generate a zero matrix. Used to generate Accumulator test input matrices.
  FuncOp zeroAccumulatorMatrix;
  // Generate an "identity" matrix, that is, a matrix whose entries are 1 on the
  // main diagonal and 0 elsewhere. Used to generate LHS and RHS test input
  // matrices.
  FuncOp identityMatrix;
  // Generate an "identity" matrix. Used to generate Accumulator test input
  // matrices.
  FuncOp identityAccumulatorMatrix;
  // Generate a pseudorandom matrix. Used to generate LHS and RHS test input
  // matrices.
  FuncOp randomMatrix;
  // Generate a pseudorandom matrix. Used to generate Accumulator test input
  // matrices.
  FuncOp randomAccumulatorMatrix;
};

// Shape of a matrix multiplication.
// The left-hand side ("LHS") matrix has shape    m x k.
// The right-hand side ("RHS") matrix has shape   k x n.
// The accumulator and result matrices have shape m x n.
struct MatmulShape {
  int m;
  int k;
  int n;
};

// Enum indexing into the functions provided by StandardFuncOps to generate
// LHS/RHS input matrices.
enum class MatrixGenerator {
  kZero,
  kIdentity,
  kRandom,
};

// Enum indexing into the functions provided by StandardFuncOps to generate
// accumulator input matrices.
enum class AccumulatorMatrixGenerator {
  kZero,
  kIdentity,
  kRandom,
};

// Enum selecting whether we're going to test dynamically shaped matrices (like
// tensor<?x?xf32>) or statically shaped matrices (like tensor<100x100xf32>).
enum class ShapeDynamicity {
  // Every dimension is dynamic.
  kDynamic,
  // Every dimension is static.
  kStatic,
  // Each dimension of each matrix is randomly picked either dynamic or static.
  kRandomlyMixed,
};

// Parameters for a test-case besides MatmulShape.
// A (MatmulGenerator, MatmulShape) pair fully specifies a test-case.
struct MatmulGenerator {
  MatrixGenerator lhs;
  MatrixGenerator rhs;
  AccumulatorMatrixGenerator acc;
  ShapeDynamicity dynamicity;
};

std::string str(MatrixGenerator generator) {
  switch (generator) {
    case MatrixGenerator::kZero:
      return "zero";
    case MatrixGenerator::kIdentity:
      return "identity";
    case MatrixGenerator::kRandom:
      return "random";
  }
}

std::string str(AccumulatorMatrixGenerator generator) {
  switch (generator) {
    case AccumulatorMatrixGenerator::kZero:
      return "zero_accumulator";
    case AccumulatorMatrixGenerator::kIdentity:
      return "identity_accumulator";
    case AccumulatorMatrixGenerator::kRandom:
      return "random_accumulator";
  }
}

std::string str(ShapeDynamicity dynamicity) {
  switch (dynamicity) {
    case ShapeDynamicity::kDynamic:
      return "dynamic_shapes";
    case ShapeDynamicity::kStatic:
      return "static_shapes";
    case ShapeDynamicity::kRandomlyMixed:
      return "randomly_mixed_shapes";
  }
}

FuncOp func(const StandardFuncOps& standardFuncOps, MatrixGenerator generator) {
  switch (generator) {
    case MatrixGenerator::kZero:
      return standardFuncOps.zeroMatrix;
    case MatrixGenerator::kIdentity:
      return standardFuncOps.identityMatrix;
    case MatrixGenerator::kRandom:
      return standardFuncOps.randomMatrix;
  }
}

FuncOp func(const StandardFuncOps& standardFuncOps,
            AccumulatorMatrixGenerator generator) {
  switch (generator) {
    case AccumulatorMatrixGenerator::kZero:
      return standardFuncOps.zeroAccumulatorMatrix;
    case AccumulatorMatrixGenerator::kIdentity:
      return standardFuncOps.identityAccumulatorMatrix;
    case AccumulatorMatrixGenerator::kRandom:
      return standardFuncOps.randomAccumulatorMatrix;
  }
}

std::vector<MatmulShape> GetTestShapes() {
  return {// Small sizes, square matrices
          {1, 1, 1},
          {2, 2, 2},
          {3, 3, 3},
          {4, 4, 4},
          {5, 5, 5},
          {8, 8, 8},
          {9, 9, 9},
          // Small sizes, slightly rectangular matrices
          {2, 3, 4},
          {8, 7, 6},
          {15, 16, 17},
          // Small sizes, involving vectors (i.e. most rectangular cases)
          {10, 1, 1},
          {1, 10, 1},
          {1, 1, 10},
          {1, 10, 10},
          {10, 1, 10},
          {10, 10, 1},
          // Small sizes, involving other very small dimensions just above 1
          {13, 14, 2},
          // Large sizes, square matrices
          {100, 100, 100},
          // Large sizes, slightly rectangular matrices
          {101, 102, 103},
          // Large sizes, involving vectors (i.e. most rectangular cases)
          {10000, 1, 1},
          {1, 10000, 1},
          {1, 1, 10000},
          {1, 1000, 1000},
          {1000, 1, 1000},
          {1000, 1000, 1},
          // Large sizes, involving other very small dimensions just above 1
          {1300, 1300, 2},
          {1300, 1300, 3},
          {1300, 1300, 4},
          // Large sizes, involving powers of two
          {256, 256, 512},
          {512, 512, 128},
          // Large sizes, involving powers of two minus one
          {127, 63, 511},
          {1023, 127, 31},
          // Large sizes, involving powers of two plus one
          {129, 65, 512},
          {1025, 129, 33}};
  // NOTE: when adding more large sizes here, please be mindful of
  // the impact on test latencies. We might have to add higher-latency test
  // cases when we implement complex logic for dealing with large matmuls.
  // At that point, it might make sense to split such test cases into
  // separate tests.
}

std::vector<MatmulGenerator> GetTestGenerators() {
  // The approach here, of unconditionally testing a variety of special
  // matrix forms before going to the general case where all matrices are
  // random, is only a compromise between latency and code complexity.
  // When the general random case succeeds, clearly all the special cases
  // also succeed, so they are redundant. They are only useful to get
  // easier-to-understand failures, in the failure case. Therefore, when
  // we will need to optimize test latencies, we will want to consider
  // implementing logic of the form
  //
  //   // Compute and compare general random matmul case, but do not
  //   // use check.expect for now because we do not want logging to
  //   // start with a hard-to-debug random-matrix test case log.
  //   if (general random matmul fails) {
  //     Perform and check.expect the normal sequence of tests
  //     involving special matrices.
  //   }
  //
  // This conditional logic would need to be in the generated code,
  // which is why we haven't thought it worth the complexity for now.
  return {
      // Start with the easiest case, Identity*Identity+Zero, that already
      // catches 90% of bugs in practice.
      {MatrixGenerator::kIdentity, MatrixGenerator::kIdentity,
       AccumulatorMatrixGenerator::kZero, ShapeDynamicity::kDynamic},
      // Then test variants where each of the three matrices (lhs, rhs, acc)
      // is a general pseudorandom matrix while the other two are still as
      // above.
      {MatrixGenerator::kRandom, MatrixGenerator::kIdentity,
       AccumulatorMatrixGenerator::kZero, ShapeDynamicity::kDynamic},
      {MatrixGenerator::kIdentity, MatrixGenerator::kRandom,
       AccumulatorMatrixGenerator::kZero, ShapeDynamicity::kDynamic},
      {MatrixGenerator::kIdentity, MatrixGenerator::kIdentity,
       AccumulatorMatrixGenerator::kRandom, ShapeDynamicity::kDynamic},
      // Then test the general case where all 3 matrices are random.
      // This catches the most bugs, but when such a test failure occurs,
      // it's the hardest to reason about.
      {MatrixGenerator::kRandom, MatrixGenerator::kRandom,
       AccumulatorMatrixGenerator::kRandom, ShapeDynamicity::kDynamic},
      // Then test variants involving static shapes.
      {MatrixGenerator::kRandom, MatrixGenerator::kRandom,
       AccumulatorMatrixGenerator::kRandom, ShapeDynamicity::kStatic},
      {MatrixGenerator::kRandom, MatrixGenerator::kRandom,
       AccumulatorMatrixGenerator::kRandom, ShapeDynamicity::kRandomlyMixed},
  };
}

// Loads the collection of helper functions that we expect to be present
// in any module that this pass is applied to.
StandardFuncOps loadStandardFuncOps(ModuleOp& moduleOp) {
  auto findFunc = [&](const char* sym_name) -> FuncOp {
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (funcOp.sym_name() == sym_name) {
        return funcOp;
      }
    }
    llvm::errs() << "No function named `" << sym_name << "` in module.\n";
    return FuncOp();
  };

  // TODO: this has quadradic complexity in the number of functions being
  // loaded. This doesn't matter for now (done once on pass initialization, and
  // small number).
  StandardFuncOps standardFuncOps;
  standardFuncOps.referenceMatmul = findFunc("reference_matmul");
  standardFuncOps.zeroMatrix = findFunc("zero_matrix");
  standardFuncOps.randomMatrix = findFunc("random_matrix");
  standardFuncOps.identityMatrix = findFunc("identity_matrix");
  standardFuncOps.zeroAccumulatorMatrix = findFunc("zero_accumulator_matrix");
  standardFuncOps.randomAccumulatorMatrix =
      findFunc("random_accumulator_matrix");
  standardFuncOps.identityAccumulatorMatrix =
      findFunc("identity_accumulator_matrix");
  return standardFuncOps;
}

// Returns a Value containing a matrix returned by the funcOp generator
// function, which takes (rows, cols) arguments and may additionally
// take a seed argument in the case of a random matrix generator.
Value createMatrix(Location loc, Value rows, Value cols, FuncOp funcOp,
                   OpBuilder& builder, std::minstd_rand& randomEngine) {
  if (funcOp.getType().getInputs().size() == 2) {
    // If funcOp takes 2 arguments, they are rows and cols.
    auto callOp = builder.create<CallOp>(loc, funcOp, ValueRange{rows, cols});
    return callOp.getResults()[0];
  } else {
    // funcOp takes 3 arguments: rows, cols, seed.
    assert(funcOp.getType().getInputs().size() == 3);
    // Generate a random seed value.
    // We don't use std::random_*_distribution because they are
    // implementation-defined, resulting in inconsistent test failures across
    // standard library implementations. We're assuming here that randomEngine
    // returns uint_fast32_t values, per spec. Let's guard that assumption e.g.
    // in case we swap random engines in the future.
    static_assert(std::is_same<decltype(randomEngine()), uint_fast32_t>::value,
                  "");
    uint32_t seed_u32 = randomEngine();
    // make a random i32 value out of a random u32 value. Can be done in
    // signed arithmetic but it's tricky enough to avoid undefined behavior
    // (have to go to i64 to perform the arithmetic to put the value
    // into i32 range) that it's just simpler to do with a memcpy.
    // TODO: in c++20 this will be a std::bit_cast.
    int32_t seed_i32;
    std::memcpy(&seed_i32, &seed_u32, sizeof seed_i32);
    // Make a Value and call the function.
    Value seed =
        builder.create<ConstantOp>(loc, builder.getI32IntegerAttr(seed_i32));
    auto callOp =
        builder.create<CallOp>(loc, funcOp, ValueRange{rows, cols, seed});
    return callOp.getResults()[0];
  }
}

// Performs the 'actual' matmul i.e. the one exercising the code that we aim
// to provide test coverage for, and that will be compared against the
// reference matmul.
//
// If we were testing only dynamic shapes, this function would be just
// creating the linalg::MatmulOp.
//
// Everything else than that is to generate test cases exercising static
// and mixed shapes. We do that by generating tensor::CastOp's casting the
// input tensors from dynamic to static shapes, performing the MatmulOp
// on those, then casting the result back to dynamic-shape.
Value actualMatmul(Location loc, OpBuilder& builder, Value lhs, Value rhs,
                   Value acc, MatmulShape shape, ShapeDynamicity dynamicity,
                   std::minstd_rand& randomEngine) {
  // Returns a boolean telling whether to use a fixed size.
  auto decideWhetherToUseFixedSize = [&]() -> bool {
    switch (dynamicity) {
      case ShapeDynamicity::kDynamic:
        return false;
      case ShapeDynamicity::kStatic:
        return true;
      case ShapeDynamicity::kRandomlyMixed: {
        // We don't use std::random_*_distribution because they are
        // implementation-defined, resulting in inconsistent test failures
        // across standard library implementations. We bother about using the
        // highest bit, not the lowest bit, not because we need high quality
        // random numbers but out of fear that the lowest bit would have a very
        // low period cycle/pattern that would limit our coverage of all cases.
        assert(randomEngine.min() == 0 || randomEngine.min() == 1);
        return randomEngine() > randomEngine.max() / 2;
      }
    }
  };
  // Generates a tensor.cast, unless the matrix already has the requested
  // tensor shape.
  auto castIfNeeded = [&](Value matrix, int64_t dstShapeRows,
                          int64_t dstShapeCols) -> Value {
    auto rankedTensorType = matrix.getType().cast<RankedTensorType>();
    auto shape = rankedTensorType.getShape();
    if (shape[0] == dstShapeRows && shape[1] == dstShapeCols) {
      return matrix;
    }
    auto elementType = rankedTensorType.getElementType();
    auto castDstType =
        mlir::RankedTensorType::get({dstShapeRows, dstShapeCols}, elementType);
    return builder.create<tensor::CastOp>(loc, castDstType, matrix).dest();
  };
  // Wrap a matrix as needed to pass it to the matmul, given the
  // wanted dynamicity.
  auto wrap = [&](Value matrix, int rows, int cols) -> Value {
    int64_t staticShapeRows =
        decideWhetherToUseFixedSize() ? rows : mlir::ShapedType::kDynamicSize;
    int64_t staticShapeCols =
        decideWhetherToUseFixedSize() ? cols : mlir::ShapedType::kDynamicSize;
    return castIfNeeded(matrix, staticShapeRows, staticShapeCols);
  };
  Value wrapLhs = wrap(lhs, shape.m, shape.k);
  Value wrapRhs = wrap(rhs, shape.k, shape.n);
  Value wrapAcc = wrap(acc, shape.m, shape.n);
  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc, ValueRange{wrapLhs, wrapRhs}, wrapAcc);
  Value matmulResult = matmulOp.getResults()[0];
  return castIfNeeded(matmulResult, mlir::ShapedType::kDynamicSize,
                      mlir::ShapedType::kDynamicSize);
};

void createEntryPointMatmulTestFunc(OpBuilder& moduleBuilder, ModuleOp moduleOp,
                                    MatmulShape shape,
                                    MatmulGenerator generator,
                                    StandardFuncOps& standardFuncOps,
                                    std::minstd_rand& randomEngine) {
  Location loc = moduleOp.getLoc();
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());

  // Create a `() -> ()` entry point op the MatmulTest tool can run.
  std::string funcName = llvm::formatv(
      "{0}_{1}_{4}x{5}_times_{2}_{5}x{6}_plus_{3}", str(generator.dynamicity),
      str(generator.lhs), str(generator.rhs), str(generator.acc), shape.m,
      shape.k, shape.n);
  auto funcOp = moduleBuilder.create<FuncOp>(
      loc, funcName, moduleBuilder.getFunctionType({}, {}));
  funcOp.setPublic();
  Block* block = funcOp.addEntryBlock();
  auto builder = OpBuilder::atBlockBegin(block);

  // Wraps a value in a DoNotOptimizeOp.
  auto doNotOptimize = [&](Value val) -> Value {
    return builder.create<IREE::Util::DoNotOptimizeOp>(loc, val).results()[0];
  };

  // Wrapping the constant size values in DoNotOptimize ensures that
  // our dynamic shapes are not turned into static shapes by the compiler
  // due to constants-propagation. We explicitly test both dynamic and static
  // shapes and want to control that.
  Value mConstOp = doNotOptimize(
      builder.create<ConstantOp>(loc, builder.getIndexAttr(shape.m)));
  Value kConstOp = doNotOptimize(
      builder.create<ConstantOp>(loc, builder.getIndexAttr(shape.k)));
  Value nConstOp = doNotOptimize(
      builder.create<ConstantOp>(loc, builder.getIndexAttr(shape.n)));

  // Select generator functions to use to generate the matmul test input
  // matrices.
  FuncOp lhsFunc = func(standardFuncOps, generator.lhs);
  FuncOp rhsFunc = func(standardFuncOps, generator.rhs);
  FuncOp accFunc = func(standardFuncOps, generator.acc);

  // Generate the matmul test inputs.
  Value lhs =
      createMatrix(loc, mConstOp, kConstOp, lhsFunc, builder, randomEngine);
  Value rhs =
      createMatrix(loc, kConstOp, nConstOp, rhsFunc, builder, randomEngine);
  Value acc =
      createMatrix(loc, mConstOp, nConstOp, accFunc, builder, randomEngine);

  // Perform the actual matmul -- exercise the code that we want to test.
  auto actualResult = actualMatmul(loc, builder, lhs, rhs, acc, shape,
                                   generator.dynamicity, randomEngine);

  // Perform the reference matmul, obtaining the reference to compare the actual
  // results to.
  auto referenceCallOp = builder.create<CallOp>(
      loc, standardFuncOps.referenceMatmul, ValueRange{lhs, rhs, acc});
  Value referenceResult = referenceCallOp.getResults()[0];

  // The comparision of actual vs reference results depends on whether the
  // element type is integral.
  bool isInteger = actualResult.getType()
                       .cast<RankedTensorType>()
                       .getElementType()
                       .isIntOrIndex();
  if (isInteger) {
    builder.create<IREE::Check::ExpectEqOp>(loc, actualResult, referenceResult);
  } else {
    // TODO: we will eventually want more control over the fuzzy comparison
    // here.
    builder.create<IREE::Check::ExpectAlmostEqOp>(loc, actualResult,
                                                  referenceResult);
  }

  builder.create<mlir::ReturnOp>(loc);
}

}  // namespace

// Clones each exported functions (including those just created) with
// placeholder constant inputs instead of arguments and removes the exported
// attribute from the old functions.
// The input are provided using util.globals.
class ExportMatmulTestFuncsPass
    : public ExportMatmulTestFuncsBase<ExportMatmulTestFuncsPass> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<IREE::Check::CheckDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder moduleBuilder(&getContext());
    StandardFuncOps standardFuncOps = loadStandardFuncOps(moduleOp);
    for (auto shape : GetTestShapes()) {
      for (auto generator : GetTestGenerators()) {
        createEntryPointMatmulTestFunc(moduleBuilder, moduleOp, shape,
                                       generator, standardFuncOps,
                                       randomEngine);
      }
    }
  }

 private:
  // std::minstd_rand is used because it's fully specified.
  // std::default_random_engine is implementation-defined.
  // Tests must use fully specified pseudorandom numbers to avoid
  // portability issues, such as test failures when compiled against other
  // C++ standard libraries.
  std::minstd_rand randomEngine;
};

std::unique_ptr<OperationPass<ModuleOp>> createExportMatmulTestFuncsPass() {
  return std::make_unique<ExportMatmulTestFuncsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
