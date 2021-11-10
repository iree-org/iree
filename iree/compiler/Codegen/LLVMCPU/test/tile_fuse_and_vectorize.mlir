// RUN: iree-opt %s -iree-llvmcpu-tile-fuse-and-vectorize -cse -canonicalize -split-input-file | IreeFileCheck %s

#config = #iree_codegen.lowering.config<tile_sizes = [[], [32, 32, 32], [16, 16, 16]], native_vector_size = [16, 16, 16]>
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func @dot_384x512x128_dispatch_0() {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c384 = arith.constant 384 : index
    %c128 = arith.constant 128 : index
    %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:384x512xf32>
    %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:512x128xf32>
    %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:384x128xf32>
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
        %10 = linalg.fill(%cst, %7) : f32, tensor<64x64xf32> -> tensor<64x64xf32> 
        %11 = linalg.matmul {lowering.config = #config} ins(%8, %9 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
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
  hal.interface private @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 64)>
//      CHECK: func @dot_384x512x128_dispatch_0() {
//  CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG: %[[CST_VECTOR:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
//  CHECK-DAG: %[[C384:.+]] = arith.constant 384 : index
//  CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
//  CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
//  CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
//      CHECK: %[[LHS:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:384x512xf32>
//      CHECK: %[[RHS:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:512x128xf32>
//      CHECK: %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:384x128xf32>
//      CHECK: %[[DST_TILE_INIT:.+]] = linalg.init_tensor
//      CHECK: scf.for %[[I_IDX:.+]] = {{.*}} to %[[C384]] step %{{[0-9]*}} {
//      CHECK:   %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]], {{.*}} -> tensor<64x512xf32>
//      CHECK:   scf.for %[[J_IDX:.+]] = {{.*}} to %[[C128]] step %{{[0-9]*}} {
//      CHECK:     %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]], {{.*}} -> tensor<512x64xf32>
//      CHECK:     {{.*}} = scf.for %[[L1_I:.+]] = %[[C0]] to %[[C64]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ITER0:.+]] = %[[DST_TILE_INIT]]) -> (tensor<64x64xf32>)
//      CHECK:       {{.*}} = scf.for %[[L1_J:.+]] = %[[C0]] to %[[C64]] step %[[C32]]
// CHECK-SAME:         iter_args(%[[ITER1:.+]] = %[[ITER0]]) -> (tensor<64x64xf32>)
//      CHECK:         %[[SLICE1:.+]] = tensor.extract_slice %[[ITER1]][%[[L1_I]], %[[L1_J]]] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
//      CHECK:         {{.*}} = scf.for %[[VEC_I:.+]] = %[[C0]] to %[[C32]] step %[[C16]]
// CHECK-SAME:           iter_args(%[[ITER2:.+]] = %[[SLICE1]]) -> (tensor<32x32xf32>)
//      CHECK:           {{.*}} = scf.for %[[VEC_J:.+]] = %[[C0]] to %[[C32]] step %[[C16]]
// CHECK-SAME:             iter_args(%[[ITER3:.+]] = %[[ITER2]]) -> (tensor<32x32xf32>)
//      CHECK:         %[[MATMUL_RES:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C512]] step %[[C16]]
// CHECK-SAME:           iter_args(%[[ITER2:.+]] = %[[CST_VECTOR]]) -> (vector<16x16xf32>)
//      CHECK:           {{.*}} = vector.transfer_read %[[LHS_TILE]]
//      CHECK:           {{.*}} = vector.transfer_read %[[RHS_TILE]]
// CHECK-COUNT-16:       vector.outerproduct
//      CHECK:           scf.yield %{{.*}} : vector<16x16xf32>
//      CHECK:         %[[EXP:.+]] = math.exp %[[MATMUL_RES]] : vector<16x16xf32>
//      CHECK:         %[[RES:.+]] = vector.transfer_write %[[EXP]], %[[ITER3]][%[[VEC_I]], %[[VEC_J]]] {{.*}} : vector<16x16xf32>, tensor<32x32xf32>
//      CHECK:         scf.yield %[[RES]]
