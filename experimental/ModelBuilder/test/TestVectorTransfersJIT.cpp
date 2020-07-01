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

// clang-format off

// NOLINTNEXTLINE
// RUN: test-vector-transfers-jit -runtime-support=$(dirname %s)/runtime-support.so | IreeFileCheck %s

// clang-format on

#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

static llvm::cl::opt<std::string> runtimeSupport(
    "runtime-support", llvm::cl::desc("Runtime support library filename"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

void TestVectorTransfers(ArrayRef<int64_t> szA, ArrayRef<int64_t> szB,
                         ArrayRef<int64_t> szVec, ArrayRef<int64_t> shA,
                         ArrayRef<int64_t> shBWr, ArrayRef<int64_t> shBRd) {
  assert(szA.size() == shA.size());
  assert(szB.size() == shBWr.size());
  assert(szB.size() == shBRd.size());
  assert(szVec.size() <= shA.size());
  assert(szVec.size() <= shBWr.size());
  assert(szVec.size() <= shBRd.size());
  ModelBuilder mb;
  // Build a func "vector_transfers".
  constexpr StringLiteral funcName = "vector_transfers";
  auto func = mb.makeFunction(funcName, {}, {},
                              MLIRFuncOpConfig().setEmitCInterface(true));
  OpBuilder b(&func.getBody());
  ScopedContext scope(b, func.getLoc());

  SmallVector<Value, 4> indicesA;
  indicesA.reserve(szA.size());
  for (auto s : shA) indicesA.push_back(std_constant_index(s));
  SmallVector<Value, 4> indicesBWr;
  indicesBWr.reserve(szB.size());
  for (auto s : shBWr) indicesBWr.push_back(std_constant_index(s));
  SmallVector<Value, 4> indicesBRd;
  indicesBRd.reserve(indicesBRd.size());
  for (auto s : shBRd) indicesBRd.push_back(std_constant_index(s));

  // clang-format off
  MLIRContext *ctx = mb.getContext();
  Value flt_0 = mb.constant_f32(0.0f);
  Value flt_1 = mb.constant_f32(1.0f);
  Value flt_42 = mb.constant_f32(42.0f);

  Value A = std_alloc(mb.getMemRefType(szA, mb.f32));
  Value B = std_alloc(mb.getMemRefType(szB, mb.f32));
  linalg_fill(A, flt_0);
  linalg_fill(B, flt_1);

  Value vFullA = vector_transfer_read(
      mb.getVectorType(szA, mb.f32),
      A,
      SmallVector<Value, 4>(szA.size(), std_constant_index(0)),
      AffineMap::getMinorIdentityMap(szA.size(), szA.size(), ctx),
      flt_42,
      ArrayAttr());
  Value vA = vector_transfer_read(
      mb.getVectorType(szVec, mb.f32),
      A,
      indicesA,
      AffineMap::getMinorIdentityMap(szA.size(), szVec.size(), ctx),
      flt_42,
      ArrayAttr());

  auto mapB = AffineMap::getMinorIdentityMap(szB.size(), szVec.size(), ctx);
  vector_transfer_write(vA, B, indicesBWr, mapB);

  Value flt_13 = std_constant_float(APFloat(13.0f), mb.f32);
  Value vFullB = vector_transfer_read(
      mb.getVectorType(szB, mb.f32),
      B,
      SmallVector<Value, 4>(szB.size(), std_constant_index(0)),
      AffineMap::getMinorIdentityMap(szB.size(), szB.size(), ctx),
      flt_13,
      ArrayAttr());
  Value vB = vector_transfer_read(
      mb.getVectorType(szVec, mb.f32),
      B,
      indicesBRd,
      AffineMap::getMinorIdentityMap(szB.size(), szVec.size(), ctx),
      flt_13,
      ArrayAttr());

  (vector_print(vFullA));
  (vector_print(vA));
  (vector_print(vFullB));
  (vector_print(vB));

  (std_dealloc(A));
  (std_dealloc(B));
  std_ret();
  // clang-format on

  // Compile the function, pass in runtime support library
  //    to the execution engine for vector.print.
  ModelRunner runner(mb.getModuleRef());
  runner.compile(CompilationOptions(), runtimeSupport);

  // Call the funcOp.
  auto err = runner.invoke(funcName);
  if (err) llvm_unreachable("Error running function.");
}

int main(int argc, char **argv) {
  ModelBuilder::registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestVectorTransfers\n");

  // A[5]
  // CHECK: ( 0, 0, 0, 0, 0 )
  // Load v4 from A[5]@{3:7} with 42 padding
  // CHECK: ( 0, 0, 42, 42 )
  // Store v4 into B[6]@{1:5}
  // CHECK: ( 1, 0, 0, 42, 42, 1 )
  // Load v4 from B[6]@{4:8} with 13 padding
  // CHECK: ( 42, 1, 13, 13 )
  TestVectorTransfers(
      /*szA=*/{5}, /*szB=*/{6}, /*szVec=*/{4}, /*shA=*/{3}, /*shBWr=*/{1},
      /*shBRd=*/{4});

  // A[5]
  // CHECK: ( 0, 0, 0, 0, 0 )
  // Load v4 from A[5]@{3:7} with 42 padding
  // CHECK: ( 0, 0, 42, 42 )
  // Store v4 into B[7]@{4:8} don't write out of bounds
  // CHECK: ( 1, 1, 1, 1, 0, 0, 42 )
  // Load v4 from B[7]@{5:9} with 13 padding
  // CHECK: ( 0, 42,  13, 13 )
  TestVectorTransfers(
      /*szA=*/{5}, /*szB=*/{7}, /*szVec=*/{4}, /*shA=*/{3}, /*shBWr=*/{4},
      /*shBRd=*/{5});

  // A[2][5]
  //      CHECK: ( ( 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0 ),
  // CHECK-SAME: ( 0, 0, 0, 0, 0 ) )
  // Load v4 from A[4][5]@{3:3}{3:7} with 42 padding
  // CHECK: ( 0, 0, 42, 42 )
  // Store v4 into B[6]@{1:5}
  // CHECK: ( 1, 0, 0, 42, 42, 1 )
  // Load v4 from B[6]@{3:7} with 13 padding
  // CHECK: ( 42, 42, 1, 13 )
  TestVectorTransfers(
      /*szA=*/{4, 5}, /*szB=*/{6}, /*szVec=*/{4}, /*shA=*/{3, 3}, /*shBWr=*/{1},
      /*shBRd=*/{3});

  // A[3][4]
  //      CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // Load v2x3 from A[3][4]@{2:4}{3:6} with 42 padding
  //      CHECK: ( ( 0, 42, 42 ), ( 42, 42, 42 ) )
  // Store v2x3 into B[4][5]@{1:3}{3:6}
  //      CHECK: ( ( 1, 1, 1, 1, 1 ), ( 1, 1, 1, 0, 42 ),
  // CHECK-SAME: ( 1, 1, 1, 42, 42 ), ( 1, 1, 1, 1, 1 ) )
  // Load v2x3 from B[4][5]@{0:2}{3:6} with 13 padding
  //      CHECK: ( ( 1, 1, 13 ), ( 0, 42, 13 ) )
  TestVectorTransfers(
      /*szA=*/{3, 4}, /*szB=*/{4, 5}, /*szVec=*/{2, 3}, /*shA=*/{2, 3},
      /*shBWr=*/{1, 3},
      /*shBRd=*/{0, 3});
}
