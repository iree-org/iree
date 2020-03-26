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
static AffineMap makeMap(ModelBuilder &mb, int i, int j) {
  SmallVector<AffineExpr, 4> results;
  results.push_back(getAffineDimExpr(i, mb.getContext()));
  results.push_back(getAffineDimExpr(j, mb.getContext()));
  return AffineMap::get(3, 0, results);
}

// Helper method to build a NxN matrix-matrix-transpose multiplication function
// using vector dialect that runs I times to amortize any calling overhead.
template <unsigned N, unsigned I>
void buildMatMat(ModelBuilder &mb, StringLiteral fn) {
  auto f32 = mb.f32;
  auto nnVectorType = mb.getVectorType({N, N}, f32);
  auto typeA = mb.getMemRefType({}, nnVectorType);
  auto typeB = typeA;
  auto typeC = typeA;

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

  // Loop I times over the kernel to amortize calling overhead.
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
template <unsigned N, bool MeasureBuild>
void BM_MxMT_UsingVector(benchmark::State &state) {
  // Prepare arguments beforehand.
  auto oneInit = [](unsigned idx, Vector2D<N, N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < N * N; ++i) p[i] = 1.0f;
  };
  auto incInit = [](unsigned idx, Vector2D<N, N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < N * N; ++i) p[i] = 1.0f + i;
  };
  auto zeroInit = [](unsigned idx, Vector2D<N, N, float> *ptr) {
    float *p = reinterpret_cast<float *>(ptr + idx);
    for (unsigned i = 0; i < N * N; ++i) p[i] = 0.0f;
  };
  auto A = makeInitializedStridedMemRefDescriptor<Vector2D<N, N, float>, 1>(
      {1}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<Vector2D<N, N, float>, 1>(
      {1}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<Vector2D<N, N, float>, 1>(
      {1}, zeroInit);
  auto *bufferA = A.get();
  auto *bufferB = B.get();
  auto *bufferC = C.get();
  void *args[3] = {&bufferA, &bufferB, &bufferC};
  StringLiteral funcName = "mat_mul_trans";
  const std::string kFuncAdapterName =
      (llvm::Twine("_mlir_ciface_") + funcName).str();

  if (MeasureBuild) {
    // If this is a build-time benchmark, build, compile, and execute
    // the function inside the timed loop, building a fresh new function
    // in each iteration to get the full JIT time (keep I == 1 here).
    for (auto _ : state) {
      ModelBuilder builder;
      buildMatMat<N, 1>(builder, funcName);
      ModelRunner runner(builder.getModuleRef());
      runner.compile(CompilationOptions());
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
    buildMatMat<N, 1000>(builder, funcName);
    ModelRunner runner(builder.getModuleRef());
    runner.compile(CompilationOptions());
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
// Benchmark drivers (build and run).
//

#define JIT true
#define RUN false
#define BENCHMARK_MAT_MUL_TRANS(SZ_N)                 \
  BENCHMARK_TEMPLATE(BM_MxMT_UsingVector, SZ_N, JIT); \
  BENCHMARK_TEMPLATE(BM_MxMT_UsingVector, SZ_N, RUN);

BENCHMARK_MAT_MUL_TRANS(1);
BENCHMARK_MAT_MUL_TRANS(2);
BENCHMARK_MAT_MUL_TRANS(4);
BENCHMARK_MAT_MUL_TRANS(8);
BENCHMARK_MAT_MUL_TRANS(16);
