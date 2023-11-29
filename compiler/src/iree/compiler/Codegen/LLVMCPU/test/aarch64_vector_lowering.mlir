// RUN: iree-opt %s --iree-llvmcpu-mmt4d-vector-lowering --split-input-file | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @dot_384x512x128_dispatch_0() {
    %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c384 = arith.constant 384 : index
    %c128 = arith.constant 128 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %3 = affine.apply #map0()[%workgroup_id_y]
    %4 = affine.apply #map0()[%workgroup_count_y]
    %5 = affine.apply #map0()[%workgroup_id_x]
    %6 = affine.apply #map0()[%workgroup_count_x]
    %7 = tensor.empty() : tensor<64x64xf32>
    scf.for %arg0 = %3 to %c384 step %4 {
      %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [64, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<64x512xf32>
      scf.for %arg1 = %5 to %c128 step %6 {
        %9 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [512, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x64xf32>
        %10 = scf.for %arg2 = %c0 to %c64 step %c16 iter_args(%arg3 = %7) -> (tensor<64x64xf32>) {
          %11 = scf.for %arg4 = %c0 to %c64 step %c16 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
            %12 = scf.for %arg6 = %c0 to %c512 step %c32 iter_args(%arg7 = %cst) -> (vector<16x16xf32>) {
              %15 = vector.transfer_read %8[%arg2, %arg6], %cst_0 {in_bounds = [true, true]} : tensor<64x512xf32>, vector<16x32xf32>
              %16 = vector.transfer_read %9[%arg6, %arg4], %cst_0 {in_bounds = [true, true]} : tensor<512x64xf32>, vector<32x16xf32>
              %17 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %16, %arg7 : vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>
              scf.yield %17 : vector<16x16xf32>
            }
            %13 = math.exp %12 : vector<16x16xf32>
            %14 = vector.transfer_write %13, %arg5[%arg2, %arg4] {in_bounds = [true, true]} : vector<16x16xf32>, tensor<64x64xf32>
            scf.yield %14 : tensor<64x64xf32>
          }
          scf.yield %11 : tensor<64x64xf32>
        }
        flow.dispatch.tensor.store %10, %2, offsets = [%arg0, %arg1], sizes = [%c64, %c64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
      }
    }
    return
  }
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 64)>
//      CHECK: func.func @dot_384x512x128_dispatch_0() {
//  CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[CST_VECTOR:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
//  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG: %[[C384:.+]] = arith.constant 384 : index
//  CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
//  CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
//  CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
//      CHECK: %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
//      CHECK: %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
//      CHECK: %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
//      CHECK: %[[DST_TILE_INIT:.+]] = tensor.empty()
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

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
module {
  func.func @matmul_gather() {
    %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c384 = arith.constant 384 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e-01 : f32
    %cst_2 = arith.constant 4.000000e-01 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %c1835008 = arith.constant 1835008 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x384xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) offset(%c1835008) : !flow.dispatch.tensor<readonly:tensor<2x512xf32>>
    %5 = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
    %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %7 = affine.apply #map0()[%workgroup_id_y]
    %8 = affine.apply #map0()[%workgroup_count_y]
    %9 = affine.apply #map0()[%workgroup_id_x]
    %10 = affine.apply #map0()[%workgroup_count_x]
    %11 = tensor.empty() : tensor<64x64xf32>
    %12 = tensor.empty() : tensor<32x32xf32>
    scf.for %arg0 = %7 to %c384 step %8 {
      %13 = flow.dispatch.tensor.load %0, offsets = [%arg0], sizes = [64], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xi32>> -> tensor<64xi32>
      %14 = flow.dispatch.tensor.load %2, offsets = [%arg0, 0], sizes = [64, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x384xf32>> -> tensor<64x384xf32>
      scf.for %arg1 = %9 to %c512 step %10 {
        %15 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<64x64xf32>
        %16 = flow.dispatch.tensor.load %3, offsets = [0, %arg1], sizes = [384, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x64xf32>
        %17 = scf.for %arg2 = %c0 to %c64 step %c32 iter_args(%arg3 = %11) -> (tensor<64x64xf32>) {
          %18 = tensor.extract_slice %13[%arg2] [32] [1] : tensor<64xi32> to tensor<32xi32>
          %19 = scf.for %arg4 = %c0 to %c64 step %c32 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
            %20 = scf.for %arg6 = %c0 to %c384 step %c32 iter_args(%arg7 = %cst) -> (vector<32x32xf32>) {
              %26 = vector.transfer_read %14[%arg2, %arg6], %cst_0 {in_bounds = [true, true]} : tensor<64x384xf32>, vector<32x32xf32>
              %27 = vector.transfer_read %16[%arg6, %arg4], %cst_0 {in_bounds = [true, true]} : tensor<384x64xf32>, vector<32x32xf32>
              %28 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %27, %arg7 : vector<32x32xf32>, vector<32x32xf32> into vector<32x32xf32>
              scf.yield %28 : vector<32x32xf32>
            }
            %21 = vector.transfer_write %20, %12[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, tensor<32x32xf32>
            %22 = tensor.extract_slice %15[%arg2, %arg4] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %23 = tensor.extract_slice %arg5[%arg2, %arg4] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %24 = linalg.generic {indexing_maps = [#map4, #map5, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%21, %18, %22 : tensor<32x32xf32>, tensor<32xi32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
            ^bb0(%arg6: f32, %arg7: i32, %arg8: f32, %arg9: f32):
              %26 = linalg.index 1 : index
              %27 = affine.apply #map6(%arg1, %26, %arg4)
              %28 = arith.index_cast %arg7 : i32 to index
              %29 = tensor.extract %6[%28, %27] : tensor<2x512xf32>
              %30 = arith.addf %arg6, %cst_1 : f32
              %31 = arith.addf %30, %29 : f32
              %32 = arith.addf %31, %arg8 : f32
              %33 = arith.mulf %32, %cst_2 : f32
              %34 = arith.addf %33, %cst_3 : f32
              linalg.yield %34 : f32
            } -> tensor<32x32xf32>
            %25 = tensor.insert_slice %24 into %arg5[%arg2, %arg4] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
            scf.yield %25 : tensor<64x64xf32>
          }
          scf.yield %19 : tensor<64x64xf32>
        }
        flow.dispatch.tensor.store %17, %5, offsets = [%arg0, %arg1], sizes = [%c64, %c64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
      }
    }
    return
  }
}
//      CHECK: func.func @matmul_gather() {
// Check that matmul is lowered to vector ops
//      CHECK:   vector.outerproduct
//      CHECK:   linalg.generic
