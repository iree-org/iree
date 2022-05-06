// RUN: iree-opt %s --iree-llvmcpu-tile-fuse-and-vectorize --cse -canonicalize --split-input-file | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[], [16, 16, 32], [16, 16, 16]]>
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func.func @dot_384x512x128_dispatch_0() {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c384 = arith.constant 384 : index
    %c128 = arith.constant 128 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:512x128xf32>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:384x128xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %3 = affine.apply #map0()[%workgroup_id_y]
    %4 = affine.apply #map0()[%workgroup_count_y]
    scf.for %arg0 = %3 to %c384 step %4 {
      %5 = affine.apply #map0()[%workgroup_id_x]
      %6 = affine.apply #map0()[%workgroup_count_x]
      scf.for %arg1 = %5 to %c128 step %6 {
        %7 = linalg.init_tensor [64, 64] : tensor<64x64xf32>
        %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [64, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<64x512xf32>
        %9 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [512, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x128xf32> -> tensor<512x64xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %11 = linalg.matmul {lowering_config = #config} ins(%8, %9 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %12 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<64x64xf32>) outs(%7 : tensor<64x64xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
          %13 = math.exp %arg2 : f32
          linalg.yield %13 : f32
        } -> tensor<64x64xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1], sizes = [%c64, %c64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:384x128xf32>
      }
    }
    return
  }
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 64)>
//      CHECK: func @dot_384x512x128_dispatch_0() {
//  CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[CST_VECTOR:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
//  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG: %[[C384:.+]] = arith.constant 384 : index
//  CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
//  CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
//  CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
//      CHECK: %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
//      CHECK: %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:512x128xf32>
//      CHECK: %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:384x128xf32>
//      CHECK: %[[DST_TILE_INIT:.+]] = linalg.init_tensor
//      CHECK: scf.for %[[I_IDX:.+]] = {{.*}} to %[[C384]] step %{{[0-9]*}} {
//      CHECK:   %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]], {{.*}} -> tensor<64x512xf32>
//      CHECK:   scf.for %[[J_IDX:.+]] = {{.*}} to %[[C128]] step %{{[0-9]*}} {
//      CHECK:     %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]], {{.*}} -> tensor<512x64xf32>
//      CHECK:     {{.*}} = scf.for %[[L1_I:.+]] = %[[C0]] to %[[C64]] step %[[C16]]
// CHECK-SAME:       iter_args(%[[ITER0:.+]] = %[[DST_TILE_INIT]]) -> (tensor<64x64xf32>)
//      CHECK:       {{.*}} = scf.for %[[L1_J:.+]] = %[[C0]] to %[[C64]] step %[[C16]]
// CHECK-SAME:         iter_args(%[[ITER1:.+]] = %[[ITER0]]) -> (tensor<64x64xf32>)
//      CHECK:         %[[MATMUL_RES:.+]] = scf.for %[[L1_K:.+]] = %[[C0]] to %[[C512]] step %[[C32]]
// CHECK-SAME:           iter_args(%[[ITER2:.+]] = %[[CST_VECTOR]]) -> (vector<16x16xf32>)
//      CHECK:           {{.*}} = vector.transfer_read %[[LHS_TILE]]
//      CHECK:           {{.*}} = vector.transfer_read %[[RHS_TILE]]
// CHECK-COUNT-16:       vector.outerproduct
// CHECK-COUNT-16:       vector.outerproduct
//      CHECK:           scf.yield %{{.*}} : vector<16x16xf32>
//      CHECK:         %[[EXP:.+]] = math.exp %[[MATMUL_RES]] : vector<16x16xf32>
//      CHECK:         %[[RES:.+]] = vector.transfer_write %[[EXP]], %[[ITER1]][%[[L1_I]], %[[L1_J]]] {{.*}} : vector<16x16xf32>, tensor<64x64xf32>
//      CHECK:         scf.yield %[[RES]]

// -----

func.func @matmul_gather() {
  %c512 = arith.constant 512 : index
  %c384 = arith.constant 384 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e-01 : f32
  %cst_1 = arith.constant 4.000000e-01 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %c1835008 = arith.constant 1835008 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:384xi32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x384xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
  %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) offset(%c1835008) : !flow.dispatch.tensor<readonly:2x512xf32>
  %5 = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) : !flow.dispatch.tensor<writeonly:384x512xf32>
  %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:2x512xf32> -> tensor<2x512xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %7 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %8 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  scf.for %arg0 = %7 to %c384 step %8 {
    %9 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %10 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    scf.for %arg1 = %9 to %c512 step %10 {
      %11 = flow.dispatch.tensor.load %0, offsets = [%arg0], sizes = [64], strides = [1] : !flow.dispatch.tensor<readonly:384xi32> -> tensor<64xi32>
      %12 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<64x64xf32>
      %13 = linalg.init_tensor [64, 64] : tensor<64x64xf32>
      %14 = flow.dispatch.tensor.load %2, offsets = [%arg0, 0], sizes = [64, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:384x384xf32> -> tensor<64x384xf32>
      %15 = flow.dispatch.tensor.load %3, offsets = [0, %arg1], sizes = [384, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<384x64xf32>
      %16 = linalg.init_tensor [64, 64] : tensor<64x64xf32>
      %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %18 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[], [32, 32, 32], [16, 16, 16]]>} ins(%14, %15 : tensor<64x384xf32>, tensor<384x64xf32>) outs(%17 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %11, %12 : tensor<64x64xf32>, tensor<64xi32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) {
      ^bb0(%arg2: f32, %arg3: i32, %arg4: f32, %arg5: f32):  // no predecessors
        %20 = linalg.index 1 : index
        %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%20, %arg1)
        %22 = arith.index_cast %arg3 : i32 to index
        %23 = tensor.extract %6[%22, %21] : tensor<2x512xf32>
        %24 = arith.addf %arg2, %cst_0 : f32
        %25 = arith.addf %24, %23 : f32
        %26 = arith.addf %25, %arg4 : f32
        %27 = arith.mulf %26, %cst_1 : f32
        %28 = arith.addf %27, %cst_2 : f32
        linalg.yield %28 : f32
      } -> tensor<64x64xf32>
      flow.dispatch.tensor.store %19, %5, offsets = [%arg0, %arg1], sizes = [%c64, %c64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:384x512xf32>
    }
  }
  return
}

//      CHECK: func @matmul_gather() {
// Check that matmul is lowered to vector ops
//  CHECK-NOT:   linalg.matmul
//      CHECK:   vector.outerproduct
// Check that generic op is still there since gather is not vectorizable.
//      CHECK:   linalg.generic

// -----

func.func @nonvectorizable_matmul_and_vectorizable_generic() {
  %c96 = arith.constant 96 : index
  %c784 = arith.constant 784 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e-03 : f32
  %cst_1 = arith.constant 3.000000e+00 : f32
  %cst_2 = arith.constant 6.000000e+00 : f32
  %cst_3 = arith.constant 0.166666672 : f32
  %c384 = arith.constant 384 : index
  %c1152 = arith.constant 1152 : index
  %c768 = arith.constant 768 : index
  %c0 = arith.constant 0 : index
  %c49 = arith.constant 49 : index
  %c16 = arith.constant 16 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c768) : !flow.dispatch.tensor<readonly:96xf32>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:96xf32>
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c1152) : !flow.dispatch.tensor<readonly:96xf32>
  %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c384) : !flow.dispatch.tensor<readonly:96xf32>
  %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:784x24xf32>
  %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:24x96xf32>
  %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:784x96xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_y]
  %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_y]
  scf.for %arg0 = %7 to %c784 step %8 {
    %9 = affine.apply affine_map<()[s0] -> (s0 * 49)>()[%workgroup_id_x]
    %10 = affine.apply affine_map<()[s0] -> (s0 * 49)>()[%workgroup_count_x]
    scf.for %arg1 = %9 to %c96 step %10 {
      %11 = flow.dispatch.tensor.load %0, offsets = [%arg1], sizes = [49], strides = [1] : !flow.dispatch.tensor<readonly:96xf32> -> tensor<49xf32>
      %12 = flow.dispatch.tensor.load %1, offsets = [%arg1], sizes = [49], strides = [1] : !flow.dispatch.tensor<readonly:96xf32> -> tensor<49xf32>
      %13 = flow.dispatch.tensor.load %2, offsets = [%arg1], sizes = [49], strides = [1] : !flow.dispatch.tensor<readonly:96xf32> -> tensor<49xf32>
      %14 = flow.dispatch.tensor.load %3, offsets = [%arg1], sizes = [49], strides = [1] : !flow.dispatch.tensor<readonly:96xf32> -> tensor<49xf32>
      %15 = linalg.init_tensor [16, 49] : tensor<16x49xf32>
      %16 = flow.dispatch.tensor.load %4, offsets = [%arg0, 0], sizes = [16, 24], strides = [1, 1] : !flow.dispatch.tensor<readonly:784x24xf32> -> tensor<16x24xf32>
      %17 = flow.dispatch.tensor.load %5, offsets = [0, %arg1], sizes = [24, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:24x96xf32> -> tensor<24x49xf32>
      %18 = linalg.init_tensor [16, 49] : tensor<16x49xf32>
      %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<16x49xf32>) -> tensor<16x49xf32>
      %20 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[], [16, 16, 32], [16, 16, 16]]>} ins(%16, %17 : tensor<16x24xf32>, tensor<24x49xf32>) outs(%19 : tensor<16x49xf32>) -> tensor<16x49xf32>
      %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %11, %12, %13, %14 : tensor<16x49xf32>, tensor<49xf32>, tensor<49xf32>, tensor<49xf32>, tensor<49xf32>) outs(%15 : tensor<16x49xf32>) {
      ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32):  // no predecessors
        %22 = arith.addf %arg5, %cst_0 : f32
        %23 = math.sqrt %22 : f32
        %24 = arith.subf %arg2, %arg3 : f32
        %25 = arith.mulf %24, %arg4 : f32
        %26 = arith.divf %25, %23 : f32
        %27 = arith.addf %26, %arg6 : f32
        %28 = arith.addf %27, %cst_1 : f32
        %29 = arith.minf %28, %cst_2 : f32
        %30 = arith.maxf %29, %cst : f32
        %31 = arith.mulf %30, %cst_3 : f32
        %32 = arith.mulf %31, %27 : f32
        linalg.yield %32 : f32
      } -> tensor<16x49xf32>
      flow.dispatch.tensor.store %21, %6, offsets = [%arg0, %arg1], sizes = [%c16, %c49], strides = [1, 1] : tensor<16x49xf32> -> !flow.dispatch.tensor<writeonly:784x96xf32>
    }
  }
  return
}

// CHECK: func @nonvectorizable_matmul_and_vectorizable_generic
// Verify that both matmul and generic ops are not vectorized.
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[C49:.+]] = arith.constant 49 : index
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         %{{.+}} = scf.for %{{.*}} = %[[C0]] to %[[C49]] step %[[C16]]
// CHECK:           %{{.+}} = linalg.fill
// CHECK:           %{{.+}} = scf.for
// CHECK:             %{{.+}} = scf.for
// CHECK:               linalg.matmul
// CHECK:           linalg.generic
