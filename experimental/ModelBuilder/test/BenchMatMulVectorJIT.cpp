// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: bench-matmul-vector-jit --benchmark_filter=all

#include "benchmark/benchmark.h"
#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;  // NOLINT

// Helper method to construct an affine map.
static AffineMap makeMap(ModelBuilder &mb, int i, int j) {
  SmallVector<AffineExpr, 4> results;
  results.push_back(getAffineDimExpr(i, mb.getContext()));
  results.push_back(getAffineDimExpr(j, mb.getContext()));
  return AffineMap::get(3, 0, results);
}

// Helper method to build matrix-matrix-transposed multiplication.
template <unsigned M, unsigned N, unsigned K, unsigned I>
void buildMatMat(ModelBuilder &mb, StringLiteral fn) {
  auto f32 = mb.f32;
  auto mkVectorType = mb.getVectorType({M, K}, f32);
  auto typeA = mb.getMemRefType({}, mkVectorType);
  auto knVectorType = mb.getVectorType({K, N}, f32);
  auto typeB = mb.getMemRefType({}, knVectorType);
  auto mnVectorType = mb.getVectorType({M, N}, f32);
  auto typeC = mb.getMemRefType({}, mnVectorType);

  auto f = mb.makeFunction(fn, {}, {typeA, typeB, typeC});
  OpBuilder b(&f.getBody());
  ScopedContext scope(b, f.getLoc());

  // Build the following accesses:
  //   affine_map<(i, j, k) -> (i, k)>,
  //   affine_map<(i, j, k) -> (j, k)>,
  //   affine_map<(i, j, k) -> (i, j)>
  SmallVector<AffineMap, 4> accesses;
  accesses.push_back(makeMap(mb, 0, 2));
  accesses.push_back(makeMap(mb, 1, 2));
  accesses.push_back(makeMap(mb, 0, 1));

  // Build the following iterator types:
  //   iterator_types = ["parallel", "parallel", "reduction"]
  SmallVector<Attribute, 4> iterator_types;
  iterator_types.push_back(mb.getStringAttr("parallel"));
  iterator_types.push_back(mb.getStringAttr("parallel"));
  iterator_types.push_back(mb.getStringAttr("reduction"));

  // Loop I times over the kernel to reduce the JIT's overhead.
  auto loop =
      b.create<loop::ForOp>(f.getLoc(), std_constant_index(0),
                            std_constant_index(I), std_constant_index(1));

  OpBuilder bodyBuilder = loop.getBodyBuilder();
  {
    edsc::ScopedContext bodyScope(bodyBuilder, f.getLoc());
    // Compute C += A x B^T with row-wise dot-products.
    StdIndexedValue A(f.getArgument(0)), B(f.getArgument(1)),
        C(f.getArgument(2));
    C() = (vector_contract(*A(), *B(), *C(), mb.getAffineMapArrayAttr(accesses),
                           mb.getArrayAttr(iterator_types)));
  }

  std_ret();
}

// Benchmark method.
template <unsigned M, unsigned N, unsigned K>
void testMatMulUsingVectors(benchmark::State &state, StringLiteral funcName,
                            bool measureBuild) {
  // Prepare arguments beforehand.
  auto oneInit = [](unsigned idx, Vector2D<M, N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < M * N; ++i) p[i] = 1.0f;
  };
  auto incInit = [](unsigned idx, Vector2D<M, N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < M * N; ++i) p[i] = 1.0f + i;
  };
  auto zeroInit = [](unsigned idx, Vector2D<M, N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < M * N; ++i) p[i] = 0.0f;
  };
  auto A = makeInitializedStridedMemRefDescriptor<Vector2D<M, N, float>, 1>(
      {1}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<Vector2D<M, N, float>, 1>(
      {1}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<Vector2D<M, N, float>, 1>(
      {1}, zeroInit);
  auto *bufferA = A.get();
  auto *bufferB = B.get();
  auto *bufferC = C.get();
  void *args[3] = {&bufferA, &bufferB, &bufferC};
  const std::string kFuncAdapterName =
      (llvm::Twine("_mlir_ciface_") + funcName).str();

  if (measureBuild) {
    // If this is a build-time benchmark, build, compile, and execute
    // the function inside the timed loop, building a fresh new function
    // in each iteration to get the full JIT time (keep I == 1 here).
    for (auto _ : state) {
      ModelBuilder builder;
      buildMatMat<M, N, K, 1>(builder, funcName);
      ModelRunner runner(builder.getModuleRef());
      runner.compile(/*llvmOptLevel=*/3, /*llcOptLevel=*/3);
      auto err = runner.engine->invoke(kFuncAdapterName,
                                       MutableArrayRef<void *>{args});
      if (err) llvm_unreachable("Error compiling/running function.");
    }
  } else {
    // If this is a run-time benchmark, build, compile, and execute
    // the function once outside the timed loop, then continue running
    // the same function inside the loop to focus on actual runtime
    // (set I == 1000 here to amortize calling overhead).
    ModelBuilder builder;
    buildMatMat<M, N, K, 1000>(builder, funcName);
    ModelRunner runner(builder.getModuleRef());
    runner.compile(/*llvmOptLevel=*/3, /*llcOptLevel=*/3);
    auto err =
        runner.engine->invoke(kFuncAdapterName, MutableArrayRef<void *>{args});
    if (err) llvm_unreachable("Error compiling/running function.");
    for (auto _ : state) {
      auto err_run = runner.engine->invoke(kFuncAdapterName,
                                           MutableArrayRef<void *>{args});
      if (err_run) llvm_unreachable("Error running function.");
    }
  }
}

//
// Benchmark drivers (build).
//

static void BM_Build_MatMul_1_1(benchmark::State &state) {
  testMatMulUsingVectors<1, 1, 1>(state, "test_matmul_1_1_1", true);
}
BENCHMARK(BM_Build_MatMul_1_1);

static void BM_Build_MatMul_2_2(benchmark::State &state) {
  testMatMulUsingVectors<2, 2, 2>(state, "test_matmul_2_2_2", true);
}
BENCHMARK(BM_Build_MatMul_2_2);

static void BM_Build_MatMul_4_4(benchmark::State &state) {
  testMatMulUsingVectors<4, 4, 4>(state, "test_matmul_4_4_4", true);
}
BENCHMARK(BM_Build_MatMul_4_4);

static void BM_Build_MatMul_8_8(benchmark::State &state) {
  testMatMulUsingVectors<8, 8, 8>(state, "test_matmul_8_8_8", true);
}
BENCHMARK(BM_Build_MatMul_8_8);

// TODO(ajcbik): enable when faster
#if 0
static void BM_Build_MatMul_16_16(benchmark::State &state) {
  testMatMulUsingVectors<16, 16, 16>(state, "test_matmul_16_16_16", true);
}
BENCHMARK(BM_Build_MatMul_16_16);
#endif

//
// Benchmark drivers (run).
//

static void BM_Run1000_MatMul_1_1(benchmark::State &state) {
  testMatMulUsingVectors<1, 1, 1>(state, "test_matmul_1_1_1", false);
}
BENCHMARK(BM_Run1000_MatMul_1_1);

static void BM_Run1000_MatMul_2_2(benchmark::State &state) {
  testMatMulUsingVectors<2, 2, 2>(state, "test_matmul_2_2_2", false);
}
BENCHMARK(BM_Run1000_MatMul_2_2);

static void BM_Run1000_MatMul_4_4(benchmark::State &state) {
  testMatMulUsingVectors<4, 4, 4>(state, "test_matmul_4_4_4", false);
}
BENCHMARK(BM_Run1000_MatMul_4_4);

static void BM_Run1000_MatMul_8_8(benchmark::State &state) {
  testMatMulUsingVectors<8, 8, 8>(state, "test_matmul_8_8_8", false);
}
BENCHMARK(BM_Run1000_MatMul_8_8);

static void BM_Run1000_MatMul_16_16(benchmark::State &state) {
  testMatMulUsingVectors<16, 16, 16>(state, "test_matmul_16_16_16", false);
}
BENCHMARK(BM_Run1000_MatMul_16_16);
