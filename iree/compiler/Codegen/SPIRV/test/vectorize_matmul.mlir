// RUN: iree-opt -split-input-file -iree-spirv-vectorize %s | IreeFileCheck %s

func @matmul_2x128x4() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<2x4xf32>
  %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<4x128xf32>
  %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<2x128xf32>
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
      %7 = memref.subview %0[%arg0, 0] [2, 4] [1, 1] : memref<2x4xf32> to memref<2x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
      %8 = memref.subview %1[0, %arg1] [4, 128] [1, 1] : memref<4x128xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
      %9 = memref.subview %2[%arg0, %arg1] [2, 128] [1, 1] : memref<2x128xf32> to memref<2x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
      %10 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %11 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %12 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%10]
      %13 = memref.subview %9[%11, %12] [1, 4] [1, 1] : memref<2x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>> to memref<1x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
      linalg.fill(%cst, %13) {__internal_linalg_transform__ = "vectorize", lowering.config = {tileSizes = [[2, 128], [], [1, 4], [0, 0, 4]]}} : f32, memref<1x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
      %17 = memref.subview %7[%11, 0] [1, 4] [1, 1] : memref<2x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>> to memref<1x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
      %18 = memref.subview %8[0, %12] [4, 4] [1, 1] : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>> to memref<4x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
      linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering.config = {tileSizes = [[2, 128], [], [1, 4], [0, 0, 4]]}}
        ins(%17, %18 : memref<1x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>, memref<4x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>)
        outs(%13 : memref<1x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>)
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

//       CHECK:   scf.for %[[IV_Y:.+]] =
//       CHECK:     %[[LHS_TILE:.+]] = memref.subview %{{.+}}[%{{.+}}, 0] [1, 4]
//       CHECK:     scf.for %[[IV_X:.+]] =
//       CHECK:       %[[ACC_TILE:.+]] = memref.subview %{{.+}}[%{{.+}}, %{{.+}}] [1, 4]
//       CHECK:       vector.transfer_write %[[ZERO]], %[[ACC_TILE]][%[[C0]], %[[C0]]]
//       CHECK:       %[[RHS_TILE:.+]] = memref.subview %{{.+}}[0, %{{.+}}] [4, 4]
//       CHECK:       %[[LHS_VECTOR:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_0_READ:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_0_T0:.+]] = vector.transpose %[[RHS_0_READ]], [1, 0]
//       CHECK:       %[[RHS_1_READ:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C1]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_1_T0:.+]] = vector.transpose %[[RHS_1_READ]], [1, 0]
//       CHECK:       %[[RHS_2_READ:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C2]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_2_T0:.+]] = vector.transpose %[[RHS_2_READ]], [1, 0]
//       CHECK:       %[[RHS_3_READ:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C3]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[RHS_3_T0:.+]] = vector.transpose %[[RHS_3_READ]], [1, 0]
//       CHECK:       %[[ACC_0:.+]] = vector.transfer_read %[[ACC_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
//       CHECK:       %[[LHS_0:.+]] = vector.extract_strided_slice %[[LHS_VECTOR]] {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
//       CHECK:       %[[RHS_0_T1:.+]] = vector.transpose %[[RHS_0_T0]], [1, 0]
//       CHECK:       %[[RHS_0_VECTOR:.+]] = vector.extract %[[RHS_0_T1]][0] : vector<1x4xf32>
//       CHECK:       %[[LHS_0_SCALAR:.+]] = vector.extract %[[LHS_0]][0, 0] : vector<1x1xf32>
//       CHECK:       %[[LHS_0_VECTOR:.+]] = splat %[[LHS_0_SCALAR]] : vector<4xf32>
//       CHECK:       %[[ACC_0_VECTOR:.+]] = vector.extract %[[ACC_0]][0] : vector<1x4xf32>
//       CHECK:       %[[FMA_0:.+]] = vector.fma %[[LHS_0_VECTOR]], %[[RHS_0_VECTOR]], %[[ACC_0_VECTOR]] : vector<4xf32>
//       CHECK:       %[[LHS_1:.+]] = vector.extract_strided_slice %[[LHS_VECTOR]] {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
//       CHECK:       %[[RHS_1_T1:.+]] = vector.transpose %[[RHS_1_T0]], [1, 0]
//       CHECK:       %[[RHS_1_VECTOR:.+]] = vector.extract %[[RHS_1_T1]][0] : vector<1x4xf32>
//       CHECK:       %[[LHS_1_SCALAR:.+]] = vector.extract %[[LHS_1]][0, 0] : vector<1x1xf32>
//       CHECK:       %[[LHS_1_VECTOR:.+]] = splat %[[LHS_1_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_1:.+]] = vector.fma %[[LHS_1_VECTOR]], %[[RHS_1_VECTOR]], %[[FMA_0]] : vector<4xf32>
//       CHECK:       %[[LHS_2:.+]] = vector.extract_strided_slice %[[LHS_VECTOR]] {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]}
//       CHECK:       %[[RHS_2_T1:.+]] = vector.transpose %[[RHS_2_T0]], [1, 0]
//       CHECK:       %[[RHS_2_VECTOR:.+]] = vector.extract %[[RHS_2_T1]][0] : vector<1x4xf32>
//       CHECK:       %[[LHS_2_SCALAR:.+]] = vector.extract %[[LHS_2]][0, 0] : vector<1x1xf32>
//       CHECK:       %[[LHS_2_VECTOR:.+]] = splat %[[LHS_2_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_2:.+]] = vector.fma %[[LHS_2_VECTOR]], %[[RHS_2_VECTOR]], %[[FMA_1]] : vector<4xf32>
//       CHECK:       %[[LHS_3:.+]] = vector.extract_strided_slice %[[LHS_VECTOR]] {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]}
//       CHECK:       %[[RHS_3_T1:.+]] = vector.transpose %[[RHS_3_T0]], [1, 0]
//       CHECK:       %[[RHS_3_VECTOR:.+]] = vector.extract %[[RHS_3_T1]][0] : vector<1x4xf32>
//       CHECK:       %[[LHS_3_SCALAR:.+]] = vector.extract %[[LHS_3]][0, 0] : vector<1x1xf32>
//       CHECK:       %[[LHS_3_VECTOR:.+]] = splat %[[LHS_3_SCALAR]] : vector<4xf32>
//       CHECK:       %[[FMA_3:.+]] = vector.fma %[[LHS_3_VECTOR]], %[[RHS_3_VECTOR]], %[[FMA_2]] : vector<4xf32>
//       CHECK:       %[[INSERT:.+]] = vector.insert %[[FMA_3]], %[[ZERO]] [0]
//       CHECK:       vector.transfer_write %[[INSERT]], %[[ACC_TILE]][%[[C0]], %[[C0]]]
