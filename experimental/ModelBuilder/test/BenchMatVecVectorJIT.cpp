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
static AffineMap makeMap(ModelBuilder &mb, int i) {
  SmallVector<AffineExpr, 4> results;
  if (i == 2) {
    results.push_back(getAffineDimExpr(0, mb.getContext()));
    results.push_back(getAffineDimExpr(1, mb.getContext()));
  } else {
    results.push_back(getAffineDimExpr(i, mb.getContext()));
  }
  return AffineMap::get(2, 0, results, mb.getContext());
}

// Helper method to build a NxN matrix-vector multiplication function
// using the vector dialect and that runs I times to amortize any calling
// overhead.
template <unsigned N, unsigned ITERS>
void buildMatMat(ModelBuilder &mb, StringLiteral fn) {
  auto f32 = mb.f32;
  auto nnVectorType = mb.getVectorType({N, N}, f32);
  auto typeA = mb.getMemRefType({}, nnVectorType);
  auto nVectorType = mb.getVectorType({N}, f32);
  auto typeB = mb.getMemRefType({}, nVectorType);
  auto typeC = typeB;

  auto f = mb.makeFunction(fn, {}, {typeA, typeB, typeC},
                           MLIRFuncOpConfig().setEmitCInterface(true));
  OpBuilder b(&f.getBody());
  ScopedContext scope(b, f.getLoc());

  // Build the following accesses:
  //   affine_map<(i, j) -> (i, j)>,
  //   affine_map<(i, j) -> (j)>,
  //   affine_map<(i, j) -> (i)>
  SmallVector<AffineMap, 4> accesses;
  accesses.push_back(makeMap(mb, 2));
  accesses.push_back(makeMap(mb, 1));
  accesses.push_back(makeMap(mb, 0));

  // Build the following iterator types:
  //   iterator_types = ["parallel", "reduction"]
  SmallVector<Attribute, 4> iterator_types;
  iterator_types.push_back(mb.getStringAttr("parallel"));
  iterator_types.push_back(mb.getStringAttr("reduction"));

  // Loop ITERS times over the kernel to reduce the JIT's overhead.
  StdIndexedValue A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  loopNestBuilder(std_constant_index(0), std_constant_index(ITERS),
                  std_constant_index(1), [&](Value) {
                    // Compute c += A x b.
                    C() = (vector_contract(A(), B(), C(),
                                           mb.getAffineMapArrayAttr(accesses),
                                           mb.getArrayAttr(iterator_types)));
                  });
  std_ret();
}

// Benchmark method.
template <unsigned N, bool MeasureBuild>
void BM_MxV_UsingVector(benchmark::State &state) {
  // Prepare arguments beforehand.
  auto incInit = [](unsigned idx, Vector2D<N, N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < N * N; ++i) p[i] = 1.0f + i;
  };
  auto oneInit = [](unsigned idx, Vector1D<N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < N; ++i) p[i] = 1.0f;
  };
  auto zeroInit = [](unsigned idx, Vector1D<N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < N; ++i) p[i] = 0.0f;
  };
  auto A = makeInitializedStridedMemRefDescriptor<Vector2D<N, N, float>, 1>(
      {1}, incInit);
  auto B = makeInitializedStridedMemRefDescriptor<Vector1D<N, float>, 1>(
      {1}, oneInit);
  auto C = makeInitializedStridedMemRefDescriptor<Vector1D<N, float>, 1>(
      {1}, zeroInit);
  StringLiteral funcName = "matvec_mult";

  if (MeasureBuild) {
    // If this is a build-time benchmark, build, compile, and execute
    // the function inside the timed loop, building a fresh new function
    // in each iteration to get the full JIT time (keep I == 1 here).
    for (auto _ : state) {
      ModelBuilder builder;
      buildMatMat<N, 1>(builder, funcName);
      ModelRunner runner(builder.getModuleRef());
      runner.compile(CompilationOptions());
      auto err = runner.invoke(funcName, A, B, C);
      if (err) llvm_unreachable("Error compiling/running function.");
    }
  } else {
    // If this is a run-time benchmark, build, compile, and execute
    // the function once outside the timed loop, then continue running
    // the same function inside the loop to focus on actual runtime
    // (set I == 1000 here to amortize calling overhead).
    ModelBuilder builder;
    buildMatMat<N, 1000>(builder, funcName);
    ModelRunner runner(builder.getModuleRef());
    runner.compile(CompilationOptions());
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
// Benchmark drivers (build and run).
//

#define JIT true
#define RUN false
#define BENCHMARK_MAT_VEC(SZ_N)                      \
  BENCHMARK_TEMPLATE(BM_MxV_UsingVector, SZ_N, JIT); \
  BENCHMARK_TEMPLATE(BM_MxV_UsingVector, SZ_N, RUN);

BENCHMARK_MAT_VEC(1);
BENCHMARK_MAT_VEC(2);
BENCHMARK_MAT_VEC(4);
BENCHMARK_MAT_VEC(8);
BENCHMARK_MAT_VEC(16);
BENCHMARK_MAT_VEC(32);
