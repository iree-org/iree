// RUN: iree-opt %s --outline-one-parent-loop="anchor-func=test anchor-op=scf.yield parent-loop-num=1 result-func-name=foo" | FileCheck %s
// RUN: iree-opt %s --outline-one-parent-loop="anchor-func=matmul anchor-op=vector.contract parent-loop-num=2 result-func-name=bar" | FileCheck %s --check-prefix=MATMUL

// CHECK-LABEL: func.func @foo
// CHECK-LABEL: func.func @test
func.func @test(%ub: index, %it: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res = scf.for %i = %c0 to %ub step %c1 iter_args(%bbit = %it) -> (index) {
    scf.yield %bbit : index
  }
  return %res: index
}

// MATMUL-LABEL: func.func @bar
// MATMUL-LABEL: func.func @matmul
func.func @matmul(%arg0: tensor<24x48xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg1: tensor<48x32xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg2: tensor<24x32xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> tensor<24x32xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c24 = arith.constant 24 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %c48 = arith.constant 48 : index
  %0 = tensor.empty() : tensor<2x2x8x32xf32>
  %1 = tensor.cast %0 : tensor<2x2x8x32xf32> to tensor<?x?x8x32xf32>
  %2 = tensor.empty() : tensor<2x2x32x8xf32>
  %3 = tensor.cast %2 : tensor<2x2x32x8xf32> to tensor<?x?x32x8xf32>
  %4 = scf.for %arg3 = %c0 to %c24 step %c16 iter_args(%arg4 = %arg2) -> (tensor<24x32xf32>) {
    %5 = affine.min affine_map<(d0) -> (16, -d0 + 24)>(%arg3)
    %6 = scf.for %arg5 = %c0 to %c32 step %c16 iter_args(%arg6 = %arg4) -> (tensor<24x32xf32>) {
      %7 = tensor.extract_slice %arg6[%arg3, %arg5] [%5, 16] [1, 1] : tensor<24x32xf32> to tensor<?x16xf32>
      %8 = scf.for %arg7 = %c0 to %5 step %c8 iter_args(%arg8 = %7) -> (tensor<?x16xf32>) {
        %13 = affine.min affine_map<(d0, d1) -> (8, -d0 + d1)>(%arg7, %5)
        %14 = scf.for %arg9 = %c0 to %c16 step %c8 iter_args(%arg10 = %arg8) -> (tensor<?x16xf32>) {
          %15 = tensor.extract_slice %arg10[%arg7, %arg9] [%13, 8] [1, 1] : tensor<?x16xf32> to tensor<?x8xf32>
          %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x8xf32>) -> tensor<?x8xf32>
          %17 = tensor.insert_slice %16 into %arg10[%arg7, %arg9] [%13, 8] [1, 1] : tensor<?x8xf32> into tensor<?x16xf32>
          scf.yield %17 : tensor<?x16xf32>
        }
        scf.yield %14 : tensor<?x16xf32>
      }
      %9 = scf.for %arg7 = %c0 to %5 step %c8 iter_args(%arg8 = %1) -> (tensor<?x?x8x32xf32>) {
        %13 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg7)
        %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg7, %arg3)
        %15 = affine.min affine_map<(d0, d1) -> (8, -d0 + d1)>(%arg7, %5)
        %16 = scf.for %arg9 = %c0 to %c48 step %c32 iter_args(%arg10 = %arg8) -> (tensor<?x?x8x32xf32>) {
          %17 = affine.apply affine_map<(d0) -> (d0 ceildiv 32)>(%arg9)
          %18 = affine.min affine_map<(d0) -> (32, -d0 + 48)>(%arg9)
          %19 = tensor.extract_slice %arg0[%14, %arg9] [%15, %18] [1, 1] : tensor<24x48xf32> to tensor<?x?xf32>
          %20 = vector.transfer_read %19[%c0, %c0], %cst : tensor<?x?xf32>, vector<8x32xf32>
          %21 = vector.transfer_write %20, %arg10[%13, %17, %c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, tensor<?x?x8x32xf32>
          scf.yield %21 : tensor<?x?x8x32xf32>
        }
        scf.yield %16 : tensor<?x?x8x32xf32>
      }
      %10 = scf.for %arg7 = %c0 to %c16 step %c8 iter_args(%arg8 = %3) -> (tensor<?x?x32x8xf32>) {
        %13 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg7)
        %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg7, %arg5)
        %15 = scf.for %arg9 = %c0 to %c48 step %c32 iter_args(%arg10 = %arg8) -> (tensor<?x?x32x8xf32>) {
          %16 = affine.apply affine_map<(d0) -> (d0 ceildiv 32)>(%arg9)
          %17 = affine.min affine_map<(d0) -> (32, -d0 + 48)>(%arg9)
          %18 = tensor.extract_slice %arg1[%arg9, %14] [%17, 8] [1, 1] : tensor<48x32xf32> to tensor<?x8xf32>
          %19 = vector.transfer_read %18[%c0, %c0], %cst {in_bounds = [false, true]} : tensor<?x8xf32>, vector<32x8xf32>
          %20 = vector.transfer_write %19, %arg10[%13, %16, %c0, %c0] {in_bounds = [true, true]} : vector<32x8xf32>, tensor<?x?x32x8xf32>
          scf.yield %20 : tensor<?x?x32x8xf32>
        }
        scf.yield %15 : tensor<?x?x32x8xf32>
      }
      %11 = scf.for %arg7 = %c0 to %5 step %c8 iter_args(%arg8 = %8) -> (tensor<?x16xf32>) {
        %13 = affine.min affine_map<(d0, d1) -> (8, -d0 + d1)>(%arg7, %5)
        %14 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg7)
        %15 = scf.for %arg9 = %c0 to %c16 step %c8 iter_args(%arg10 = %arg8) -> (tensor<?x16xf32>) {
          %16 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg9)
          %17 = tensor.extract_slice %arg10[%arg7, %arg9] [%13, 8] [1, 1] : tensor<?x16xf32> to tensor<?x8xf32>
          %18 = vector.transfer_read %17[%c0, %c0], %cst {in_bounds = [false, true]} : tensor<?x8xf32>, vector<8x8xf32>
          %19 = scf.for %arg11 = %c0 to %c48 step %c32 iter_args(%arg12 = %18) -> (vector<8x8xf32>) {
            %22 = affine.apply affine_map<(d0) -> (d0 ceildiv 32)>(%arg11)
            %23 = vector.transfer_read %9[%14, %22, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?x8x32xf32>, vector<8x32xf32>
            %24 = vector.transfer_read %10[%16, %22, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?x32x8xf32>, vector<32x8xf32>
            %25 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %24, %arg12 : vector<8x32xf32>, vector<32x8xf32> into vector<8x8xf32>
            scf.yield %25 : vector<8x8xf32>
          }
          %20 = vector.transfer_write %19, %17[%c0, %c0] {in_bounds = [false, true]} : vector<8x8xf32>, tensor<?x8xf32>
          %21 = tensor.insert_slice %20 into %arg10[%arg7, %arg9] [%13, 8] [1, 1] : tensor<?x8xf32> into tensor<?x16xf32>
          scf.yield %21 : tensor<?x16xf32>
        }
        scf.yield %15 : tensor<?x16xf32>
      }
      %12 = tensor.insert_slice %11 into %arg6[%arg3, %arg5] [%5, 16] [1, 1] : tensor<?x16xf32> into tensor<24x32xf32>
      scf.yield %12 : tensor<24x32xf32>
    }
    scf.yield %6 : tensor<24x32xf32>
  }
  return %4 : tensor<24x32xf32>
}
func.func private @nano_time() -> i64 attributes {llvm.emit_c_interface}
func.func public @main(%arg0: tensor<24x48xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg1: tensor<48x32xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg2: tensor<24x32xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}, %arg3: memref<?xi64>) -> tensor<24x32xf32> attributes {llvm.emit_c_interface} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg3, %c0 : memref<?xi64>
  %1 = scf.for %arg4 = %c0 to %0 step %c1 iter_args(%arg5 = %arg2) -> (tensor<24x32xf32>) {
    %2 = func.call @nano_time() : () -> i64
    %3 = func.call @matmul(%arg0, %arg1, %arg5) : (tensor<24x48xf32>, tensor<48x32xf32>, tensor<24x32xf32>) -> tensor<24x32xf32>
    %4 = func.call @nano_time() : () -> i64
    %5 = arith.subi %4, %2 : i64
    memref.store %5, %arg3[%arg4] : memref<?xi64>
    scf.yield %3 : tensor<24x32xf32>
  }
  return %1 : tensor<24x32xf32>
}
