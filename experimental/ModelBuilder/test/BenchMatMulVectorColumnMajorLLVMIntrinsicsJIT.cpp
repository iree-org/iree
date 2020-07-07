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

#include "benchmark/benchmark.h"
#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"

using namespace mlir;  // NOLINT

// Helper method to construct an affine map.
static SmallVector<AffineMap, 3> makeColumnMajorMatmulMaps(ModelBuilder &mb) {
  AffineExpr m, n, k;
  bindDims(mb.getContext(), m, n, k);
  SmallVector<AffineMap, 3> results;
  results.push_back(AffineMap::get(3, 0, {k, n}, mb.getContext()));
  results.push_back(AffineMap::get(3, 0, {m, k}, mb.getContext()));
  results.push_back(AffineMap::get(3, 0, {n, m}, mb.getContext()));
  return results;
}

// Helper method to build a matrix-matrix column-major multiplication function
// using the vector dialect and that runs ITERS times to amortize any calling
// overhead.
template <unsigned M, unsigned N, unsigned K, unsigned ITERS>
void buildMatMat(ModelBuilder &mb, StringLiteral fn) {
  auto f32 = mb.f32;
  auto mkVectorType = mb.getVectorType({M, K}, f32);
  auto typeA = mb.getMemRefType({}, mkVectorType);
  auto knVectorType = mb.getVectorType({K, N}, f32);
  auto typeB = mb.getMemRefType({}, knVectorType);
  auto mnVectorType = mb.getVectorType({M, N}, f32);
  auto typeC = mb.getMemRefType({}, mnVectorType);

  auto f = mb.makeFunction(
      fn, {}, {typeA, typeB, typeC},
      MLIRFuncOpConfig().setEmitCInterface(true).setPreferAvx512(true));
  OpBuilder b(&f.getBody());
  ScopedContext scope(b, f.getLoc());

  // Build the following accesses:
  //   affine_map<(m, n, k) -> (k, m)>,
  //   affine_map<(m, n, k) -> (n, k)>,
  //   affine_map<(m, n, k) -> (n, m)>
  SmallVector<AffineMap, 4> accesses = makeColumnMajorMatmulMaps(mb);

  // Build the following iterator types:
  //   iterator_types = ["parallel", "parallel", "reduction"]
  SmallVector<Attribute, 4> iterator_types;
  iterator_types.push_back(mb.getStringAttr("parallel"));
  iterator_types.push_back(mb.getStringAttr("parallel"));
  iterator_types.push_back(mb.getStringAttr("reduction"));

  // Loop ITERS times over the kernel to reduce the JIT's overhead.
  StdIndexedValue A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  loopNestBuilder(std_constant_index(0), std_constant_index(ITERS),
                  std_constant_index(1), [&](Value) {
                    // Compute C += A x B, in column-major form, with LLVM
                    // matrix intrinsics.
                    C() = (vector_contract(A(), B(), C(),
                                           mb.getAffineMapArrayAttr(accesses),
                                           mb.getArrayAttr(iterator_types)));
                  });
  std_ret();
}

// Benchmark method.
template <unsigned M, unsigned N, unsigned K, bool MeasureBuild,
          bool LowerToLLVMMatrixIntrinsics>
void BM_MxMColMajorVectors(benchmark::State &state) {
  constexpr unsigned NumMxMPerIteration = 1000;
  state.counters["NumMxM/Iter"] = NumMxMPerIteration;
  // Column major vector types.
  using TypeLHS = Vector2D<K, M, float>;
  using TypeRHS = Vector2D<N, K, float>;
  using TypeRES = Vector2D<N, M, float>;
  // Prepare arguments beforehand.
  auto oneInit = [](unsigned idx, TypeLHS *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < M * N; ++i) p[i] = 1.0f;
  };
  auto incInit = [](unsigned idx, TypeRHS *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < M * N; ++i) p[i] = 1.0f + i;
  };
  auto zeroInit = [](unsigned idx, TypeRES *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < M * N; ++i) p[i] = 0.0f;
  };
  auto A = makeInitializedStridedMemRefDescriptor<TypeLHS, 1>({1}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<TypeRHS, 1>({1}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<TypeRES, 1>({1}, zeroInit);
  StringLiteral funcName = "matmult_column_major";

  vector::VectorTransformsOptions vectorTransformsOptions{
      LowerToLLVMMatrixIntrinsics ? vector::VectorContractLowering::Matmul
                                  : vector::VectorContractLowering::Dot};
  CompilationOptions compilationOptions{/*llvmOptLevel=*/3, /*llcOptLevel=*/3,
                                        vectorTransformsOptions};
  if (MeasureBuild) {
    // If this is a build-time benchmark, build, compile, and execute
    // the function inside the timed loop, building a fresh new function
    // in each iteration to get the full JIT time (keep I == 1 here).
    for (auto _ : state) {
      ModelBuilder builder;
      buildMatMat<M, N, K, 1>(builder, funcName);
      ModelRunner runner(builder.getModuleRef());
      runner.compile(compilationOptions);
      auto err = runner.invoke(funcName, A, B, C);
      if (err) llvm_unreachable("Error compiling/running function.");
    }
  } else {
    // If this is a run-time benchmark, build, compile, and execute
    // the function once outside the timed loop, then continue running
    // the same function inside the loop to focus on actual runtime
    // (set I == NumIterations here to amortize calling overhead).
    ModelBuilder builder;
    buildMatMat<M, N, K, NumMxMPerIteration>(builder, funcName);
    ModelRunner runner(builder.getModuleRef());
    runner.compile(compilationOptions);
    auto err = runner.invoke(funcName, A, B, C);
    if (err) llvm_unreachable("Error compiling/running function.");
    for (auto _ : state) {
      auto err_run = runner.invoke(funcName, A, B, C);
      if (err_run) llvm_unreachable("Error running function.");
    }
  }
}

int main(int argc, char **argv) {
  mlir::ModelBuilder::registerAllDialects();
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
}

//
// Benchmark drivers (build).
//

#define BENCHMARK_MATMUL_COLUMN_MAJOR(SZ_M, SZ_N, SZ_K)                      \
  BENCHMARK_TEMPLATE(BM_MxMColMajorVectors, SZ_M, SZ_N, SZ_K, true, false);  \
  BENCHMARK_TEMPLATE(BM_MxMColMajorVectors, SZ_M, SZ_N, SZ_K, true, true);   \
  BENCHMARK_TEMPLATE(BM_MxMColMajorVectors, SZ_M, SZ_N, SZ_K, false, false); \
  BENCHMARK_TEMPLATE(BM_MxMColMajorVectors, SZ_M, SZ_N, SZ_K, false, true);

BENCHMARK_MATMUL_COLUMN_MAJOR(1, 1, 1);
BENCHMARK_MATMUL_COLUMN_MAJOR(2, 2, 2);
BENCHMARK_MATMUL_COLUMN_MAJOR(4, 4, 4);
BENCHMARK_MATMUL_COLUMN_MAJOR(8, 8, 8);
BENCHMARK_MATMUL_COLUMN_MAJOR(16, 16, 16);
