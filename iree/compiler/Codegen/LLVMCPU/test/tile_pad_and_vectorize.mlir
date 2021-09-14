// RUN: iree-opt %s -cse -iree-llvmcpu-tile-pad-and-vectorize -cse -canonicalize -split-input-file | IreeFileCheck %s

#config0 = {tileSizes = [[64, 64]]}
#config1 = {nativeVectorSize = [4, 4, 4], tileSizes = [[64, 64], [32, 32, 32], [4, 4, 4]]}
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (64, -d0 + 383)>
#map2 = affine_map<(d0) -> (64, -d0 + 513)>
#map3 = affine_map<(d0) -> (-d0 + 383, 64)>
#map4 = affine_map<(d0) -> (-d0 + 513, 64)>
module  {
  func @dot_383x383x513_dispatch_0() {
    %c0 = constant 0 : index
    %c513 = constant 513 : index
    %c383 = constant 383 : index
    %cst = constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:383x383xf32>
    %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:383x513xf32>
    %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:383x513xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %3 = affine.apply #map0()[%workgroup_id_y]
    %4 = affine.apply #map0()[%workgroup_count_y]
    scf.for %arg0 = %3 to %c383 step %4 {
      %5 = affine.apply #map0()[%workgroup_id_x]
      %6 = affine.apply #map0()[%workgroup_count_x]
      scf.for %arg1 = %5 to %c513 step %6 {
        %7 = affine.min #map1(%arg0)
        %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 383], strides = [1, 1] : !flow.dispatch.tensor<readonly:383x383xf32> -> tensor<?x383xf32>
        %9 = affine.min #map2(%arg1)
        %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [383, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:383x513xf32> -> tensor<383x?xf32>
        %11 = affine.min #map3(%arg0)
        %12 = affine.min #map4(%arg1)
        %13 = linalg.init_tensor [%11, %12] : tensor<?x?xf32>
        %14 = linalg.fill(%cst, %13) {__internal_linalg_transform__ = "workgroup", lowering.config = #config0} : f32, tensor<?x?xf32> -> tensor<?x?xf32>
        %15 = linalg.matmul {__internal_linalg_transform__ = "workgroup", lowering.config = #config1} ins(%8, %10 : tensor<?x383xf32>, tensor<383x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:383x513xf32>
      }
    }
    return
  }
  hal.interface private @io  {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
//     CHECK: #[[MAP1:.+]] = affine_map<(d0) -> (64, -d0 + 383)>
//     CHECK: #[[MAP2:.+]] = affine_map<(d0) -> (64, -d0 + 513)>
//     CHECK: @dot_383x383x513_dispatch_0
// CHECK-DAG: %[[CST:.+]] = constant 0.000000e+00 : f32
// CHECK-DAG: %[[C0:.+]] = constant 0 : index
// CHECK-DAG: %[[C4:.+]] = constant 4 : index
// CHECK-DAG: %[[C383:.+]] = constant 383 : index
// CHECK-DAG: %[[C513:.+]] = constant 513 : index
// CHECK-DAG: %[[C32:.+]] = constant 32 : index
//     CHECK: %[[LHS:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:383x383xf32>
//     CHECK: %[[RHS:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:383x513xf32>
//     CHECK: %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:383x513xf32>
//     CHECK: scf.for %[[I_WG_IDX:.+]] = {{.*}} to %c383
//     CHECK: scf.for %[[J_WG_IDX:.+]] = {{.*}} to %c513
//     CHECK: %[[LHS_WG_TILE_DIM0:.+]] = affine.min #[[MAP1]](%[[I_WG_IDX]])
//     CHECK: %[[LHS_WG_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//     CHECK: %[[RHS_WG_TILE_DIM1:.+]] = affine.min #[[MAP2]](%[[J_WG_IDX]])
//     CHECK: %[[RHS_WG_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//     CHECK: %[[DST_WG_TILE_INIT:.+]] = linalg.init_tensor
//     CHECK: %[[DST_WG_TILE_INIT_C0:.+]] = linalg.fill(%[[CST]], %[[DST_WG_TILE_INIT]])
//     CHECK: {{.*}} = scf.for {{.*}} = %[[C0]] to %[[LHS_WG_TILE_DIM0]] step %[[C32]] iter_args(%[[DST_WG_TILE_0:.+]] = %[[DST_WG_TILE_INIT_C0]])
//     CHECK:    {{.*}} = scf.for {{.*}} = %[[C0]] to %[[RHS_WG_TILE_DIM1]] step %[[C32]] iter_args(%[[DST_WG_TILE_1:.+]] = %[[DST_WG_TILE_0]])
//     CHECK:       {{.*}} = scf.for {{.*}} = %[[C0]] to %[[C383]] step %[[C32]] iter_args(%[[DST_WG_TILE_2:.+]] = %[[DST_WG_TILE_1]])
//     CHECK:           %[[LHS_L1_TILE:.+]] = tensor.extract_slice %[[LHS_WG_TILE]]
//     CHECK:           %[[RHS_L1_TILE:.+]] = tensor.extract_slice %[[RHS_WG_TILE]]
//     CHECK:           %[[DST_L1_TILE:.+]] = tensor.extract_slice %[[DST_WG_TILE_2]]
//     CHECK:           %[[LHS_L1_TILE_PADDED:.+]] = linalg.pad_tensor %[[LHS_L1_TILE]]
//     CHECK:           %[[RHS_L1_TILE_PADDED:.+]] = linalg.pad_tensor %[[RHS_L1_TILE]]
//     CHECK:           %[[DST_L1_TILE_PADDED:.+]] = linalg.pad_tensor %[[DST_L1_TILE]]
//     CHECK:           {{.*}} = scf.for {{.*}} = %[[C0]] to %[[C32]] step %[[C4]] iter_args(%[[DST_VEC_TILE_0:.+]] = %[[DST_L1_TILE_PADDED]])
//     CHECK:              {{.*}} = scf.for {{.*}} = %[[C0]] to %[[C32]] step %[[C4]] iter_args(%[[DST_VEC_TILE_1:.+]] = %[[DST_VEC_TILE_0]])
//     CHECK:                {{.*}} = scf.for {{.*}} = %[[C0]] to %[[C32]] step %[[C4]] iter_args(%[[DST_VEC_TILE_2:.+]] = %[[DST_VEC_TILE_1]])
//     CHECK:                    %[[LHS_VEC_TILE:.+]] = tensor.extract_slice %[[LHS_L1_TILE_PADDED]]
//     CHECK:                    %[[RHS_VEC_TILE:.+]] = tensor.extract_slice %[[RHS_L1_TILE_PADDED]]
//     CHECK:                    %[[DST_VEC_TILE:.+]] = tensor.extract_slice %[[DST_VEC_TILE_2]]
//     CHECK:                    %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS_VEC_TILE]]
//     CHECK:                    %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS_VEC_TILE]]
//     CHECK:                    %[[DST_VEC:.+]] = vector.transfer_read %[[DST_VEC_TILE]]
