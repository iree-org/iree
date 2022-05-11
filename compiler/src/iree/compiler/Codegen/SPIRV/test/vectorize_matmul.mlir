// RUN: iree-opt --split-input-file --iree-spirv-vectorize %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 128], [1, 4], [0, 0, 4]]>

func.func @matmul_2x128x4() {
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
          %13 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%12 : tensor<1x4xf32>) -> tensor<1x4xf32>
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

//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index

//       CHECK:   scf.for %[[IV_Y:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:     %[[LHS_TILE:.+]] = tensor.extract_slice %{{.+}}[%[[IV_Y]], 0] [1, 4]
//       CHECK:     %[[LHS_VECTOR:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
//       CHECK:     scf.for %[[IV_X:.+]] = %[[C0]] to %[[C128]] step %[[C4]] iter_args(%[[ACC_TILE:.+]] =
//       CHECK:       %[[RHS_TILE:.+]] = tensor.extract_slice %{{.+}}[0, %[[IV_X]]] [4, 4]
//       CHECK:       %[[RHS_0_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_1_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C1]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_2_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C2]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_3_VECTOR:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C3]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[LHS_0_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][0]
//       CHECK:       %[[LHS_0_VECTOR:.+]] = vector.splat %[[LHS_0_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_0:.+]] = vector.fma %[[RHS_0_VECTOR]], %[[LHS_0_VECTOR]], %[[ZERO]] : vector<4xf32>
//       CHECK:       %[[LHS_1_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][1]
//       CHECK:       %[[LHS_1_VECTOR:.+]] = vector.splat %[[LHS_1_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_1:.+]] = vector.fma %[[RHS_1_VECTOR]], %[[LHS_1_VECTOR]], %[[FMA_0]] : vector<4xf32>
//       CHECK:       %[[LHS_2_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][2]
//       CHECK:       %[[LHS_2_VECTOR:.+]] = vector.splat %[[LHS_2_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_2:.+]] = vector.fma %[[RHS_2_VECTOR]], %[[LHS_2_VECTOR]], %[[FMA_1]] : vector<4xf32>
//       CHECK:       %[[LHS_3_SCALAR:.+]] = vector.extract %[[LHS_VECTOR]][3]
//       CHECK:       %[[LHS_3_VECTOR:.+]] = vector.splat %[[LHS_3_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_3:.+]] = vector.fma %[[RHS_3_VECTOR]], %[[LHS_3_VECTOR]], %[[FMA_2]] : vector<4xf32>
//       CHECK:       vector.transfer_write %[[FMA_3]], %[[ACC_TILE]][%[[IV_Y]], %[[IV_X]]]

// -----

// Check that we can vectorize shape dimensions not divisible by 4 but divisible by 2.

func.func @matmul_8x8x2(%lhs: tensor<8x2xf32>, %rhs: tensor<2x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
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

func.func @matmul_8x8x1(%lhs: tensor<8x1xf32>, %rhs: tensor<1x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs: tensor<8x1xf32>, tensor<1x8xf32>) outs(%init: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

//    CHECK-LABEL: func @matmul_8x8x1

//  CHECK-COUNT-8: vector.transfer_read {{.*}} : tensor<8x1xf32>, vector<1xf32>
//  CHECK-COUNT-2: vector.transfer_read {{.*}} : tensor<1x8xf32>, vector<4xf32>
// CHECK-COUNT-16: vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<4xf32>
// CHECK-COUNT-16: vector.fma
// CHECK-COUNT-16: vector.transfer_write {{.*}} : vector<4xf32>, tensor<8x8xf32>


// -----

func.func @matmul_broadcast_add(%init: tensor<1x8xf32>, %a: tensor<1x8xf32>, %b: tensor<8x8xf32>, %c: tensor<1x8xf32>, %bias: tensor<1xf32>) -> tensor<1x8xf32> {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index

  %matmul = linalg.matmul ins(%a, %b : tensor<1x8xf32>, tensor<8x8xf32>) outs(%c : tensor<1x8xf32>) -> tensor<1x8xf32>
  %bcast_add = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %bias : tensor<1x8xf32>, tensor<1xf32>) outs(%init : tensor<1x8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %add = arith.addf %arg0, %arg1 : f32
    linalg.yield %add : f32
  } -> tensor<1x8xf32>
  return %bcast_add: tensor<1x8xf32>
}

//    CHECK-LABEL: func @matmul_broadcast_add
//     CHECK-SAME: (%[[INIT:[a-z0-9]+]]: tensor<1x8xf32>
//     CHECK-SAME:  %[[BIAS:[a-z0-9]+]]: tensor<1xf32>)

//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index

// CHECK-COUNT-16:   vector.fma
//      CHECK-NOT:   vector.transpose

//          CHECK:   %[[READ:.+]] = vector.transfer_read %[[BIAS]]
//          CHECK:   %[[EXT0:.+]] = vector.extract %[[READ]][0] : vector<1xf32>
//          CHECK:   %[[BCST0:.+]] = vector.splat %[[EXT0]] : vector<4xf32>
//          CHECK:   %[[ADD0:.+]] = arith.addf %{{.+}}, %[[BCST0]] : vector<4xf32>
//          CHECK:   %[[EXT1:.+]] = vector.extract %[[READ]][0] : vector<1xf32>
//          CHECK:   %[[BCST1:.+]] = vector.splat %[[EXT1]] : vector<4xf32>
//          CHECK:   %[[ADD1:.+]] = arith.addf %{{.+}}, %[[BCST1]] : vector<4xf32>
//          CHECK:   %[[WRITE0:.+]] = vector.transfer_write %[[ADD0]], %[[INIT]][%[[C0]], %[[C0]]]
//          CHECK:   %[[WRITE1:.+]] = vector.transfer_write %[[ADD1]], %[[WRITE0]][%[[C0]], %[[C4]]]
//          CHECK:   return %[[WRITE1]]

// -----

func.func @matmul_2x8x128_fp16(%a: tensor<2x128xf16>, %b: tensor<128x8xf16>, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>) -> tensor<2x8xf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %f0 = arith.constant 0.0 : f16

  %init = linalg.init_tensor [2, 8] : tensor<2x8xf16>
  %fill = linalg.fill ins(%f0 : f16) outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  %matmul = scf.for %iv = %c0 to %c128 step %c8 iter_args(%arg = %fill) -> (tensor<2x8xf16>) {
    %as = tensor.extract_slice %a[0, %iv] [2, 8] [1, 1] : tensor<2x128xf16> to tensor<2x8xf16>
    %bs = tensor.extract_slice %b[%iv, 0] [8, 8] [1, 1] : tensor<128x8xf16> to tensor<8x8xf16>
    %cs = linalg.matmul ins(%as, %bs : tensor<2x8xf16>, tensor<8x8xf16>) outs(%arg : tensor<2x8xf16>) -> tensor<2x8xf16>
    scf.yield %cs : tensor<2x8xf16>
  }
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %x : tensor<2x8xf16>, tensor<2x8xf16>) outs(%y : tensor<2x8xf16>) {
  ^bb0(%arg0: f16, %arg1: f16, %arg2: f16):
    %div = arith.divf %arg0, %arg1 : f16
    linalg.yield %div : f16
  } -> tensor<2x8xf16>

  return %0: tensor<2x8xf16>
}

//    CHECK-LABEL: func @matmul_2x8x128_fp16
//     CHECK-SAME: (%{{.+}}: tensor<2x128xf16>, %{{.+}}: tensor<128x8xf16>, %[[X:.+]]: tensor<2x8xf16>, %[[Y:.+]]: tensor<2x8xf16>)
//          CHECK:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<8xf16>
//          CHECK:   %[[FOR:.+]]:2 = scf.for %arg4 = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%arg5 = %[[ZERO]], %arg6 = %[[ZERO]])
//  CHECK-COUNT-2:     vector.transfer_read {{.+}} : tensor<2x8xf16>, vector<8xf16>
//  CHECK-COUNT-8:     vector.transfer_read {{.+}} : tensor<8x8xf16>, vector<8xf16>
// CHECK-COUNT-32:     vector.fma {{.+}} : vector<4xf16>
//          CHECK:     %[[ISS0:.+]] = vector.insert_strided_slice %{{.+}}, %[[ZERO]] {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
//          CHECK:     %[[ISS1:.+]] = vector.insert_strided_slice %{{.+}}, %[[ISS0]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
//          CHECK:     %[[ISS2:.+]] = vector.insert_strided_slice %{{.+}}, %[[ZERO]] {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
//          CHECK:     %[[ISS3:.+]] = vector.insert_strided_slice %{{.+}}, %[[ISS2]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
//          CHECK:     scf.yield %[[ISS3]], %[[ISS1]] : vector<8xf16>, vector<8xf16>
//          CHECK:   }
// CHECK:   %[[X0:.+]] = vector.transfer_read %[[X]]{{.+}} : tensor<2x8xf16>, vector<8xf16>
// CHECK:   %[[X1:.+]] = vector.transfer_read %[[X]]{{.+}} : tensor<2x8xf16>, vector<8xf16>
// CHECK:   %[[LHS0:.+]] = vector.extract_strided_slice %[[FOR]]#1 {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[RHS0:.+]] = vector.extract_strided_slice %[[X0]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[DIV0:.+]] = arith.divf %[[LHS0]], %[[RHS0]]
// CHECK:   %[[ISS0:.+]] = vector.insert_strided_slice %[[DIV0]], %[[ZERO]] {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
// CHECK:   %[[LHS1:.+]] = vector.extract_strided_slice %[[FOR]]#1 {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[RHS1:.+]] = vector.extract_strided_slice %[[X0]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[DIV1:.+]] = arith.divf %[[LHS1]], %[[RHS1]]
// CHECK:   %[[ISS1:.+]] = vector.insert_strided_slice %[[DIV1]], %[[ISS0]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
// CHECK:   %[[LHS2:.+]] = vector.extract_strided_slice %[[FOR]]#0 {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[RHS2:.+]] = vector.extract_strided_slice %[[X1]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[DIV2:.+]] = arith.divf %[[LHS2]], %[[RHS2]]
// CHECK:   %[[ISS2:.+]] = vector.insert_strided_slice %[[DIV2]], %[[ZERO]] {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
// CHECK:   %[[LHS3:.+]] = vector.extract_strided_slice %[[FOR]]#0 {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[RHS3:.+]] = vector.extract_strided_slice %[[X1]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:   %[[DIV3:.+]] = arith.divf %[[LHS3]], %[[RHS3]]
// CHECK:   %[[ISS3:.+]] = vector.insert_strided_slice %[[DIV3]], %[[ISS2]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
// CHECK:   %[[W0:.+]] = vector.transfer_write %[[ISS1]], %[[Y]][%c0, %c0] {in_bounds = [true]} : vector<8xf16>, tensor<2x8xf16>
// CHECK:   %[[W1:.+]] = vector.transfer_write %[[ISS3]], %[[W0]][%c1, %c0] {in_bounds = [true]} : vector<8xf16>, tensor<2x8xf16>
// CHECK:   return %[[W1]]
