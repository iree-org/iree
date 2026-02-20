// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-opt-level=O3 --compile-to=stream %s | FileCheck %s

// No transient allocations created.
util.func @test_no_transients(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<128x512xf32> {iree.abi.output  = 0 : index}) -> tensor<128x512xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  util.return %2 : tensor<128x512xf32>
}
// CHECK-LABEL: func public @test_no_transients(
//   CHECK-NOT:   stream.resource.alloca

// -----

// No transient allocations created, and the transient size is set to 0.
util.func @test_no_transients_external_transient_buffer(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<128x512xf32> {iree.abi.output  = 0 : index}, %arg3 : !hal.buffer {iree.abi.transients}) -> tensor<128x512xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  util.return %2 : tensor<128x512xf32>
}
// CHECK-LABEL: func public @test_no_transients_external_transient_buffer(
//  CHECK-SAME:     iree.abi.transients.size = @[[SIZE_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     iree.abi.transients.size.constant = 0
//   CHECK-NOT:   stream.resource.alloca
//       CHECK: func public @[[SIZE_FN]](
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   return %[[C0]]

// -----

// Needs transient memory allocation.
util.func @test_transients(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<512x1024xf32>, %arg3 : tensor<128x1024xf32> {iree.abi.output  = 0 : index}) -> tensor<128x1024xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %3 = tensor.empty() : tensor<128x1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %5 = linalg.matmul ins(%2, %arg2 : tensor<128x512xf32>, tensor<512x1024xf32>)
      outs(%4 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  util.return %5 : tensor<128x1024xf32>
}
// CHECK-LABEL: func public @test_transients(
//       CHECK:   %[[CST:.+]] = arith.constant 262144 : index
//       CHECK:   stream.resource.alloca
//  CHECK-SAME:       !stream.resource<transient>{%[[CST]]}

// -----

// Needs transient memory allocation, but no allocas created.
util.func @test_transients_external_transient_buffer(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<512x1024xf32>, %arg3 : tensor<128x1024xf32> {iree.abi.output  = 0 : index},
    %arg4 : !hal.buffer {iree.abi.transients}) -> tensor<128x1024xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %3 = tensor.empty() : tensor<128x1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %5 = linalg.matmul ins(%2, %arg2 : tensor<128x512xf32>, tensor<512x1024xf32>)
      outs(%4 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  util.return %5 : tensor<128x1024xf32>
}
// CHECK-LABEL: func public @test_transients_external_transient_buffer(
//  CHECK-SAME:     iree.abi.transients.size = @[[SIZE_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     iree.abi.transients.size.constant = 262144
//   CHECK-NOT:   stream.resource.alloca
//       CHECK: func public @[[SIZE_FN]](
//       CHECK:   %[[CST:.+]] = arith.constant 262144 : index
//       CHECK:   return %[[CST]]
