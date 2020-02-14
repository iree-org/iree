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

// RUN: test-simple-jit 2>&1 | IreeFileCheck %s

#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;  // NOLINT

void testMatmulOnVectors() {
  constexpr unsigned K = 16, M = 4, N = 8;

  ModelBuilder modelBuilder;
  // Build a func "test_vector_matmul".
  constexpr StringLiteral funcName = "test_vector_matmul";
  auto f32 = modelBuilder.f32;
  auto mkVectorType = modelBuilder.getVectorType({M, K}, f32);
  auto typeA = modelBuilder.getMemRefType({-1, -1}, mkVectorType);
  auto knVectorType = modelBuilder.getVectorType({K, N}, f32);
  auto typeB = modelBuilder.getMemRefType({-1, -1}, knVectorType);
  auto mnVectorType = modelBuilder.getVectorType({M, N}, f32);
  auto typeC = modelBuilder.getMemRefType({-1, -1}, mnVectorType);

  // CHECK-LABEL: func @test_vector_matmul(
  //  CHECK-SAME:   %[[A:.*]]: memref<?x?xvector<4x16xf32>>,
  //  CHECK-SAME:   %[[B:.*]]: memref<?x?xvector<16x8xf32>>,
  //  CHECK-SAME:   %[[C:.*]]: memref<?x?xvector<4x8xf32>>)
  auto func = modelBuilder.makeFunction(funcName, {}, {typeA, typeB, typeC});

  OpBuilder b(&func.getBody());
  ScopedContext scope(b, func.getLoc());
  ValueHandle A(func.getArgument(0)), B(func.getArgument(1)),
      C(func.getArgument(2));
  auto contractionBuilder = [](ArrayRef<BlockArgument> args) {
    assert(args.size() == 3 && "expected 3 block arguments");
    (linalg_yield(vector_matmul(args[0], args[1], args[2])));
  };

  // Fill its body.
  //      CHECK:   linalg.generic {{.*}} %[[A]], %[[B]], %[[C]]
  //      CHECK:     vector.contract {{.*}} : vector<4x16xf32>, vector<16x8xf32>
  // CHECK-SAME:       into vector<4x8xf32>
  linalg_matmul(A, B, C, contractionBuilder);
  std_ret();

  modelBuilder.getModuleRef()->dump();
}

int main() { testMatmulOnVectors(); }
