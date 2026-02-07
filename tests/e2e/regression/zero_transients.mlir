// RUN: iree-compile --iree-hal-target-backends=rocm --iree-rocm-target=mi300x --iree-opt-level=O3 --iree-scheduling-dump-statistics-file=- --iree-scheduling-dump-statistics-format=json --compile-to=stream %s | FileCheck %s

// For this case no transient memory allocations should be required.
util.func @test_no_transients(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<128x512xf32> {iree.abi.output  = 0 : index}) -> tensor<128x512xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  util.return %2 : tensor<128x512xf32>
}
//      CHECK: transient-memory-size
// CHECK-SAME: 0
