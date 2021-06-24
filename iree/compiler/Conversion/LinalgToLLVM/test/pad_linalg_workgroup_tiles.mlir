// RUN: iree-opt %s -cse -iree-codegen-llvm-pad-linalg-workgroup-tiles -split-input-file | IreeFileCheck %s

#config0 = {tileSizes = [[64, 64]]}
#config1 = {nativeVectorSize = [4, 4, 4], tileSizes = [[64, 64], [32, 32, 32], [4, 4, 4]]}
module  {
  func @matmul_f32_5x3x5() {
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c5 = constant 5 : index
    %c64 = constant 64 : index
    %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:5x3xf32>
    %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:3x5xf32>
    %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:5x5xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %c64]
    %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %c64]
    scf.for %arg0 = %3 to %c5 step %4 {
      %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %c64]
      %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %c64]
      scf.for %arg1 = %5 to %c5 step %6 {
        %7 = affine.min affine_map<(d0) -> (64, -d0 + 5)>(%arg0)
        %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:5x3xf32> -> tensor<?x3xf32>
        %9 = affine.min affine_map<(d0) -> (64, -d0 + 5)>(%arg1)
        %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [3, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:3x5xf32> -> tensor<3x?xf32>
        %11 = affine.min affine_map<(d0) -> (64, -d0 + 5)>(%arg0)
        %12 = affine.min affine_map<(d0) -> (64, -d0 + 5)>(%arg1)
        %13 = affine.min affine_map<(d0) -> (-d0 + 5, 64)>(%arg0)
        %14 = affine.min affine_map<(d0) -> (-d0 + 5, 64)>(%arg1)
        %15 = linalg.init_tensor [%13, %14] : tensor<?x?xf32>
        %16 = linalg.fill(%cst, %15) {__internal_linalg_transform__ = "workgroup", lowering.config = #config0} : f32, tensor<?x?xf32> -> tensor<?x?xf32>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup", lowering.config = #config1} ins(%8, %10 : tensor<?x3xf32>, tensor<3x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:5x5xf32>
      }
    }
    return
  }

  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: @matmul_f32_5x3x5
//       CHECK: %[[C0:.+]] = constant 0.000000e+00 : f32
//       CHECK: %[[LHS:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external[%{{.*}}] : !flow.dispatch.tensor<readonly:5x3xf32>
//       CHECK: %[[RHS:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external[%{{.*}}] : !flow.dispatch.tensor<readonly:3x5xf32>
//       CHECK: %[[RESULT:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external[%{{.*}}] : !flow.dispatch.tensor<writeonly:5x5xf32>
//       CHECK: flow.dispatch.tensor.load %[[LHS]], offsets = [%{{.*}}, 0], sizes = [%[[LHS_TILE_SIZE:.+]], 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:5x3xf32> -> tensor<?x3xf32>
//       CHECK: flow.dispatch.tensor.load %[[RHS]], offsets = [0, %{{.*}}], sizes = [3, %[[RHS_TILE_SIZE:.+]]], strides = [1, 1] : !flow.dispatch.tensor<readonly:3x5xf32> -> tensor<3x?xf32>
//       CHECK: %[[PADDED_LHS:.+]] = linalg.pad_tensor %{{.*}} low[0, 0] high[3, 1]  {
//  CHECK-NEXT:     ^bb0(%{{.*}}: index, %{{.*}}: index):  // no predecessors
//  CHECK-NEXT: linalg.yield %[[C0]] : f32
//  CHECK-NEXT: tensor<?x3xf32> to tensor<8x4xf32>
//       CHECK: %[[PADDED_RHS:.+]] = linalg.pad_tensor %{{.*}} low[0, 0] high[1, 3]  {
//  CHECK-NEXT:     ^bb0(%{{.*}}: index, %{{.*}}: index):  // no predecessors
//  CHECK-NEXT: linalg.yield %[[C0]] : f32
//  CHECK-NEXT: tensor<3x?xf32> to tensor<4x8xf32>
//       CHECK: %[[PADDED_RESULT:.+]] = linalg.init_tensor [8, 8] : tensor<8x8xf32>
//       CHECK: %[[PADDED_RESULT_0:.+]] = linalg.fill(%[[C0]], %[[PADDED_RESULT]]) : f32, tensor<8x8xf32>
//       CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul {{.*}} ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<8x4xf32>, tensor<4x8xf32>) outs(%[[PADDED_RESULT_0]] : tensor<8x8xf32>) -> tensor<8x8xf32>
//       CHECK: %[[CLIPED_RESULT:.+]] = tensor.extract_slice %[[MATMUL_RESULT]][0, 0] [%[[LHS_TILE_SIZE]], %[[RHS_TILE_SIZE]]] [1, 1] : tensor<8x8xf32> to tensor<?x?xf32>
//       CHECK:  flow.dispatch.tensor.store %[[CLIPED_RESULT]], %[[RESULT]], offsets = [%{{.*}}, %{{.*}}], sizes = [%[[LHS_TILE_SIZE]], %[[RHS_TILE_SIZE]]], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:5x5xf32>
