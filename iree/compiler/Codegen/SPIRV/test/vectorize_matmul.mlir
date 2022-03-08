// RUN: iree-opt -split-input-file -iree-spirv-vectorize %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 128], [1, 4], [0, 0, 4]]>

func @matmul_2x128x4() {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:2x4xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:4x128xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:2x128xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_count_y]
  scf.for %arg0 = %3 to %c2 step %4 {
    %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
    %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
    scf.for %arg1 = %5 to %c128 step %6 {
      %7 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [2, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:2x4xf32> -> tensor<2x4xf32>
      %8 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [4, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:4x128xf32> -> tensor<4x128xf32>
      %9 = linalg.init_tensor [2, 128] : tensor<2x128xf32>
      %10 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %9) -> (tensor<2x128xf32>) {
        %11 = scf.for %arg4 = %c0 to %c128 step %c4 iter_args(%arg5 = %arg3) -> (tensor<2x128xf32>) {
          %12 = tensor.extract_slice %arg5[%arg2, %arg4] [1, 4] [1, 1] : tensor<2x128xf32> to tensor<1x4xf32>
          %13 = linalg.fill(%cst, %12) {lowering_config = #config} : f32, tensor<1x4xf32> -> tensor<1x4xf32>
          %14 = tensor.extract_slice %7[%arg2, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
          %15 = tensor.extract_slice %8[0, %arg4] [4, 4] [1, 1] : tensor<4x128xf32> to tensor<4x4xf32>
          %16 = linalg.matmul {lowering_config = #config} ins(%14, %15 : tensor<1x4xf32>, tensor<4x4xf32>) outs(%13 : tensor<1x4xf32>) -> tensor<1x4xf32>
          %17 = tensor.insert_slice %16 into %arg5[%arg2, %arg4] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<2x128xf32>
          scf.yield %17 : tensor<2x128xf32>
        } {iree.spirv.distribute_dim = 0 : index}
        scf.yield %11 : tensor<2x128xf32>
      } {iree.spirv.distribute_dim = 1 : index}
      flow.dispatch.tensor.store %10, %2, offsets = [%arg0, %arg1], sizes = [%c2, %c128], strides = [1, 1] : tensor<2x128xf32> -> !flow.dispatch.tensor<writeonly:2x128xf32>
    }
  }
  return
}

// CHECK-LABEL: func @matmul_2x128x4()

//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<1x4xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index

//       CHECK:   scf.for %[[IV_Y:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:     scf.for %[[IV_X:.+]] = %[[C0]] to %[[C128]] step %[[C4]] iter_args(%[[ACC_TILE:.+]] =
//       CHECK:       %[[LHS_TILE:.+]] = tensor.extract_slice %{{.+}}[%[[IV_Y]], 0] [1, 4]
//       CHECK:       %[[RHS_TILE:.+]] = tensor.extract_slice %{{.+}}[0, %[[IV_X]]] [4, 4]
//       CHECK:       %[[LHS_VECTOR:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_0_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_1_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C1]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_2_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C2]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_3_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C3]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[LHS_0_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][0]
//       CHECK:       %[[LHS_0_VECTOR:.+]] = vector.splat %[[LHS_0_SCALAR]] : vector<4xf32>
//       CHECK:       %[[ACC_0_VECTOR:.+]] = vector.extract %[[ZERO]][0] : vector<1x4xf32>
//       CHECK:       %[[FMA_0:.+]] = vector.fma %[[LHS_0_VECTOR]], %[[RHS_0_VECTOR]], %[[ACC_0_VECTOR]] : vector<4xf32>
//       CHECK:       %[[LHS_1_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][1]
//       CHECK:       %[[LHS_1_VECTOR:.+]] = vector.splat %[[LHS_1_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_1:.+]] = vector.fma %[[LHS_1_VECTOR]], %[[RHS_1_VECTOR]], %[[FMA_0]] : vector<4xf32>
//       CHECK:       %[[LHS_2_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][2]
//       CHECK:       %[[LHS_2_VECTOR:.+]] = vector.splat %[[LHS_2_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_2:.+]] = vector.fma %[[LHS_2_VECTOR]], %[[RHS_2_VECTOR]], %[[FMA_1]] : vector<4xf32>
//       CHECK:       %[[LHS_3_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][3]
//       CHECK:       %[[LHS_3_VECTOR:.+]] = vector.splat %[[LHS_3_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_3:.+]] = vector.fma %[[LHS_3_VECTOR]], %[[RHS_3_VECTOR]], %[[FMA_2]] : vector<4xf32>
//       CHECK:       vector.transfer_write %[[FMA_3]], %[[ACC_TILE]][%[[IV_Y]], %[[IV_X]]]

// -----

// Check that we can vectorize shape dimensions not divisible by 4 but divisible by 2.

func @matmul_8x8x2(%lhs: tensor<8x2xf32>, %rhs: tensor<2x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs: tensor<8x2xf32>, tensor<2x8xf32>) outs(%init: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

//    CHECK-LABEL: func @matmul_8x8x2

//  CHECK-COUNT-8: vector.transfer_read {{.*}} : tensor<8x2xf32>, vector<2xf32>
//  CHECK-COUNT-4: vector.transfer_read {{.*}} : tensor<2x8xf32>, vector<4xf32>
// CHECK-COUNT-16: vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<4xf32>
// CHECK-COUNT-16: vector.fma
// CHECK-COUNT-16: vector.transfer_write {{.*}} : vector<4xf32>, tensor<8x8xf32>

// -----

// Check that we can vectorize shape dimensions not divisible by 4/2 but divisible by 1.

func @matmul_8x8x1(%lhs: tensor<8x1xf32>, %rhs: tensor<1x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs: tensor<8x1xf32>, tensor<1x8xf32>) outs(%init: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

//    CHECK-LABEL: func @matmul_8x8x1

//  CHECK-COUNT-8: vector.transfer_read {{.*}} : tensor<8x1xf32>, vector<1xf32>
//  CHECK-COUNT-2: vector.transfer_read {{.*}} : tensor<1x8xf32>, vector<4xf32>
// CHECK-COUNT-16: vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<4xf32>
// CHECK-COUNT-16: vector.fma
// CHECK-COUNT-16: vector.transfer_write {{.*}} : vector<4xf32>, tensor<8x8xf32>
