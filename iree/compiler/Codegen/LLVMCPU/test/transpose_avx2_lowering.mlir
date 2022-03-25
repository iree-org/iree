// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=false}))' %s | FileCheck %s --check-prefix=CHECK-NON-AVX2


// CHECK-AVX2: func @test_4x8_dispatch
// CHECK-NON-AVX2: func @test_4x8_dispatch
func @test_4x8 (%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %init = linalg.init_tensor [8, 16] : tensor<8x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>
  %gemm = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %gemm : tensor<8x16xf32>
}

// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>

// CHECK-NON-AVX2-NOT: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>

// -----

// CHECK-AVX2: func @test_8x8_dispatch
// CHECK-NON-AVX2: func @test_8x8_dispatch
func @test_8x8 (%arg0 : tensor<16x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<16x16xf32> {
  %cst = arith.constant 0.0 : f32
  %init = linalg.init_tensor [16, 16] : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  %gemm = linalg.matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %gemm : tensor<16x16xf32>
}

// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK-AVX2: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>

// CHECK-NON-AVX2-NOT: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
