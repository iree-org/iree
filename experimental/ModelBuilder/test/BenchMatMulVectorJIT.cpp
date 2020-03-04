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

// RUN: true

#include "benchmark/benchmark.h"
#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;  // NOLINT

// Helper method to construct an affine map.
static AffineMap makeMap(ModelBuilder &builder, int i, int j) {
  SmallVector<AffineExpr, 4> results;
  results.push_back(getAffineDimExpr(i, builder.getContext()));
  results.push_back(getAffineDimExpr(j, builder.getContext()));
  return AffineMap::get(3, 0, results);
}

// Build matrix multiplication code and benchmark runtime.
template <unsigned M, unsigned N, unsigned K>
void testMatMulUsingVectors(benchmark::State &state, StringLiteral funcName) {
  ModelBuilder modelBuilder;

  auto f32 = modelBuilder.f32;
  auto mkVectorType = modelBuilder.getVectorType({M, K}, f32);
  auto typeA = modelBuilder.getMemRefType({}, mkVectorType);
  auto knVectorType = modelBuilder.getVectorType({K, N}, f32);
  auto typeB = modelBuilder.getMemRefType({}, knVectorType);
  auto mnVectorType = modelBuilder.getVectorType({M, N}, f32);
  auto typeC = modelBuilder.getMemRefType({}, mnVectorType);

  // 1. Build a matrix-matrix-transposed multiplication.
  {
    auto f = modelBuilder.makeFunction(funcName, {}, {typeA, typeB, typeC});
    OpBuilder b(&f.getBody());
    ScopedContext scope(b, f.getLoc());

    StdIndexedValue A(f.getArgument(0)), B(f.getArgument(1)),
        C(f.getArgument(2));

    // Build the following accesses:
    //   affine_map<(i, j, k) -> (i, k)>,
    //   affine_map<(i, j, k) -> (j, k)>,
    //   affine_map<(i, j, k) -> (i, j)>
    SmallVector<AffineMap, 4> accesses;
    accesses.push_back(makeMap(modelBuilder, 0, 2));
    accesses.push_back(makeMap(modelBuilder, 1, 2));
    accesses.push_back(makeMap(modelBuilder, 0, 1));

    // Build the following iterator types:
    //   iterator_types = ["parallel", "parallel", "reduction"]
    SmallVector<Attribute, 4> iterator_types;
    iterator_types.push_back(modelBuilder.getStringAttr("parallel"));
    iterator_types.push_back(modelBuilder.getStringAttr("parallel"));
    iterator_types.push_back(modelBuilder.getStringAttr("reduction"));

    // Compute C += A x B^T with row-wise dot-products.
    C() = (vector_contract(*A(), *B(), *C(),
                           modelBuilder.getAffineMapArrayAttr(accesses),
                           modelBuilder.getArrayAttr(iterator_types)));
    std_ret();
  }

  // 2. Compile the function. pass in runtime support library
  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(/*llvmOptLevel=*/3, /*llcOptLevel=*/3);

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
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

  // 5. Call the funcOp named `funcName`.
  const std::string kFuncAdapterName =
      (llvm::Twine("_mlir_ciface_") + funcName).str();
  auto *bufferA = A.get();
  auto *bufferB = B.get();
  auto *bufferC = C.get();
  void *args[3] = {&bufferA, &bufferB, &bufferC};
  // Call the function as many times as requested by the benchmark driver.
  // The first call is expensive, since it includes JIT time. Subsequent
  // calls are faster. By running the function multiple times, the build and
  // JIT time are amortized over the actual runtime calls.
  for (auto _ : state) {
    auto err =
        runner.engine->invoke(kFuncAdapterName, MutableArrayRef<void *>{args});
    if (err) llvm_unreachable("Error running function.");
  }
}

//
// Benchmark drivers.
//

// TODO(ntv): share one JIT between all execution engines
//            so this can run in debug mode too
#ifndef NDEBUG

static void BM_Dummy(benchmark::State &state) {
  static int stat = 0;
  for (auto _ : state) stat++;
}
BENCHMARK(BM_Dummy);

#else

static void BM_MatMul_1_1(benchmark::State &state) {
  testMatMulUsingVectors<1, 1, 1>(state, "test_matmul_1_1_1");
}
BENCHMARK(BM_MatMul_1_1);

static void BM_MatMul_2_2(benchmark::State &state) {
  testMatMulUsingVectors<2, 2, 2>(state, "test_matmul_2_2_2");
}
BENCHMARK(BM_MatMul_2_2);

static void BM_MatMul_4_4(benchmark::State &state) {
  testMatMulUsingVectors<4, 4, 4>(state, "test_matmul_4_4_4");
}
BENCHMARK(BM_MatMul_4_4);

static void BM_MatMul_8_8(benchmark::State &state) {
  testMatMulUsingVectors<8, 8, 8>(state, "test_matmul_8_8_8");
}
BENCHMARK(BM_MatMul_8_8);

static void BM_MatMul_16_16(benchmark::State &state) {
  testMatMulUsingVectors<16, 16, 16>(state, "test_matmul_16_16_16");
}
BENCHMARK(BM_MatMul_16_16);

#endif  // NDEBUG
