// RUN: iree-opt --iree-llvmcpu-lower-to-ukernels %s | FileCheck %s

func.func @mmt4d_f32f32f32(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>,
    %arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: func @mmt4d_f32f32f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 257 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[M0_index:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[M0:.+]] = arith.index_cast %[[M0_index]] : index to i32
//  CHECK-DAG:   %[[N0_index:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[N0:.+]] = arith.index_cast %[[N0_index]] : index to i32
//  CHECK-DAG:   %[[K0_index:.+]] = tensor.dim %[[ARG1]], %[[C3]]
//  CHECK-DAG:   %[[K0:.+]] = arith.index_cast %[[K0_index]] : index to i32
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[M0]], %[[N0]], %[[K0]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @mmt4d_fill(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>, %arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%fill : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: func @mmt4d_fill(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[M0_index:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[M0:.+]] = arith.index_cast %[[M0_index]] : index to i32
//  CHECK-DAG:   %[[N0_index:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[N0:.+]] = arith.index_cast %[[N0_index]] : index to i32
//  CHECK-DAG:   %[[K0_index:.+]] = tensor.dim %[[ARG1]], %[[C3]]
//  CHECK-DAG:   %[[K0:.+]] = arith.index_cast %[[K0_index]] : index to i32
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[M0]], %[[N0]], %[[K0]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @mmt4d_i8i8i32(%arg0 : tensor<?x?x?x?xi8>, %arg1 : tensor<?x?x?x?xi8>,
    %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>)
      outs(%arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}
//      CHECK: func @mmt4d_i8i8i32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi8>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 258 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[M0_index:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[M0:.+]] = arith.index_cast %[[M0_index]] : index to i32
//  CHECK-DAG:   %[[N0_index:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[N0:.+]] = arith.index_cast %[[N0_index]] : index to i32
//  CHECK-DAG:   %[[K0_index:.+]] = tensor.dim %[[ARG1]], %[[C3]]
//  CHECK-DAG:   %[[K0:.+]] = arith.index_cast %[[K0_index]] : index to i32
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[M0]], %[[N0]], %[[K0]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

//      CHECK: func @pack_i8i8(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xi8>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: i8
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 2 : i32
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[ARG2]] : i8 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "iree_uk_pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_i8i8(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?x7x8xi8>, %arg2 : i8) -> tensor<?x?x7x8xi8> {
  %result = tensor.pack %arg0 padding_value(%arg2 : i8) inner_dims_pos = [0, 1] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xi8> -> tensor<?x?x7x8xi8>
  func.return %result : tensor<?x?x7x8xi8>
}

// -----

//      CHECK: func @pack_i32i32_transpose_inner(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 259 : i32
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[ARG2]] : i32 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "iree_uk_pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_i32i32_transpose_inner(%arg0 : tensor<?x?xi32>, %arg1 : tensor<?x?x7x8xi32>, %arg2 : i32) -> tensor<?x?x7x8xi32> {
  %result = tensor.pack %arg0 padding_value(%arg2 : i32) inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xi32> -> tensor<?x?x7x8xi32>
  func.return %result : tensor<?x?x7x8xi32>
}

// -----

//      CHECK: func @pack_f32f32_transpose_inner_and_outer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: f32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 769 : i32
//  CHECK-DAG:   %[[BITCAST:.+]] = arith.bitcast %[[ARG2]] : f32 to i32
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[BITCAST]] : i32 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "iree_uk_pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_f32f32_transpose_inner_and_outer(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?x7x8xf32>, %arg2 : f32) -> tensor<?x?x7x8xf32> {
  %result = tensor.pack %arg0 padding_value(%arg2 : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xf32> -> tensor<?x?x7x8xf32>
  func.return %result : tensor<?x?x7x8xf32>
}

// -----

//      CHECK: func @unpack_i32i32_transpose_inner(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x7x8xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xi32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 258 : i32
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[IN_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[IN_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "iree_uk_unpack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[IN_SIZE2]], %[[IN_SIZE3]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[FLAGS]] :
func.func @unpack_i32i32_transpose_inner(%arg0 : tensor<?x?x7x8xi32>, %arg1 : tensor<?x?xi32>) -> tensor<?x?xi32> {
  %result = tensor.unpack %arg0 inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?x7x8xi32> -> tensor<?x?xi32>
  func.return %result : tensor<?x?xi32>
}

// -----

//      CHECK: func @unpack_f32f32_transpose_inner_and_outer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x7x8xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 769 : i32
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[IN_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[IN_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "iree_uk_unpack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[IN_SIZE2]], %[[IN_SIZE3]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[FLAGS]] :
func.func @unpack_f32f32_transpose_inner_and_outer(%arg0 : tensor<?x?x7x8xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?x7x8xf32> -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}
