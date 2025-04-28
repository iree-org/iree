// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering,canonicalize,cse))" --split-input-file %s | FileCheck %s

func.func @broadcast_read(%arg0: memref<4096x32xf16>) -> vector<1x8xf16> {
  %cst_1 = arith.constant 0.000000e+00 : f16
  %0 = gpu.thread_id  x
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %broadcast_read = vector.transfer_read %arg0[%workgroup_id_x, %0], %cst_1 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, 0)>} : memref<4096x32xf16>, vector<1x8xf16>
  return %broadcast_read : vector<1x8xf16>
}
// Check that it is just load-extract-splat-broadcast.
// CHECK-LABEL: func.func @broadcast_read
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x32xf16>)

// -----

func.func @contraction_masked(%lhs: vector<2xf16>, %rhs: vector<4x2xf16>, %acc: vector<4xf32>, %mask: vector<2x4xi1>) -> vector<4xf32> {
  %ret = vector.mask %mask { vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"], kind = #vector.kind<add>} %lhs, %rhs, %acc : vector<2xf16>, vector<4x2xf16> into vector<4xf32> } : vector<2x4xi1> -> vector<4xf32>
  return %ret: vector<4xf32>
}
// CHECK-LABEL: @contraction_masked


// -----

module {
  func.func @transfer_read_lowering(%arg0: memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, %idx : index, %mask: vector<4xi1>) -> vector<4xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %v = vector.transfer_read %arg0[%idx, %idx], %cst_0, %mask {in_bounds = [true]} : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
    return %v : vector<4xf32>
  }
}

// -----

func.func @test_gather(%cst_2 : vector<4x4xindex>, %cst_mask : vector<4x4xi1>, %cst_pt : vector<4x4xf32>, %0 : memref<32x32xf32>) -> vector<4x4xf32> {
  %c0 = arith.constant 0 : index
  %14 = vector.gather %0[%c0, %c0] [%cst_2], %cst_mask, %cst_pt : memref<32x32xf32>, vector<4x4xindex>, vector<4x4xi1>, vector<4x4xf32> into vector<4x4xf32>
  return %14 : vector<4x4xf32>
}
// CHECK-LABEL: @test_gather

// -----

func.func @multi_reduction(%arg0 : vector<2x3x4xf32>) -> vector<3xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<3xf32>
  %red = vector.multi_reduction <maximumf> , %arg0, %cst [0, 2] : vector<2x3x4xf32> to vector<3xf32>
  return %red : vector<3xf32>
}
// CHECK-LABEL: @multi_reduction

// -----

func.func @test_insert(%arg0: vector<4x1x1x1x4xf32>, %arg1: vector<4x1x1x1x4xf32>) -> vector<2x4x1x1x1x4xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<2x1x1xf16>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<2x4x1x1x1x4xf32>
  %0 = vector.insert %arg0, %cst_0 [0] : vector<4x1x1x1x4xf32> into vector<2x4x1x1x1x4xf32>
  %1 = vector.insert %arg1, %0 [1] : vector<4x1x1x1x4xf32> into vector<2x4x1x1x1x4xf32>
  return %1 : vector<2x4x1x1x1x4xf32>
}
// CHECK-LABEL: @test_insert

// -----

func.func @test_elementwise(%arg0: vector<2x3x1x1x1x4xf32>) -> vector<2x3x1x1x1x4xf16> {
  %2 = arith.truncf %arg0 : vector<2x3x1x1x1x4xf32> to vector<2x3x1x1x1x4xf16>
  return %2 : vector<2x3x1x1x1x4xf16>
}
// CHECK-LABEL: @test_elementwise

// -----

func.func @test_broadcast(%arg0 : vector<8xi32>) -> vector<3x8xi32> {
  %0 = vector.broadcast %arg0 : vector<8xi32> to vector<3x8xi32>
 return %0 : vector<3x8xi32>
}
// CHECK-LABEL: @test_broadcast

// -----

func.func @test_transpose_final_unit_dim(%arg0 : vector<2x3x1xi32>)  -> vector<3x2x1xi32> {
  %0 = vector.transpose %arg0, [1, 0, 2]: vector<2x3x1xi32> to vector<3x2x1xi32>
  return %0 : vector<3x2x1xi32>
}
// CHECK-LABEL: @test_transpose_final_unit_dim

// -----

func.func @test_transpose_no_final_leading_dim(%arg0 : vector<2x3x4xi32>)  -> vector<3x4x2xi32> {
  %0 = vector.transpose %arg0, [1, 2, 0]: vector<2x3x4xi32> to vector<3x4x2xi32>
  return %0 : vector<3x4x2xi32>
}
// CHECK-LABEL: @test_transpose_no_final_leading_dim

// -----

#broadcast_3d = affine_map<(d0, d1, d2) -> (0, d1, 0)>
func.func @transfer_broadcasting_3D(%mem : memref<8x8x8xf32>, %idx : index) -> vector<4x2x4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx, %idx], %cf0
    {in_bounds = [true, true, true], permutation_map = #broadcast_3d}
      : memref<8x8x8xf32>, vector<4x2x4xf32>
  return %res : vector<4x2x4xf32>
}
// CHECK-LABEL: func.func @transfer_broadcasting_3D
//   CHECK-NOT: vector.transfer_read
//       CHECK: return
