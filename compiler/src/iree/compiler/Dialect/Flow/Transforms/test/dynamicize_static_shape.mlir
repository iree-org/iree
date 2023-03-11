// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-flow-dynamicize-static-shapes-pass,cse))"  %s | FileCheck %s

// Check basic flow.dispatch.region structure

func.func @empty_region(%tensor: tensor<?x16x?xf32>, %dim0: index, %dim2: index) -> tensor<?x16x?xf32> {
  %c16 = arith.constant 16 : index
  %region = flow.dispatch.region[%dim0, %c16, %dim2] -> (tensor<?x16x?xf32>{%dim0, %dim2}) {
    flow.return %tensor : tensor<?x16x?xf32>
  } count(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0, %arg1, %arg2
    flow.return %x, %y, %z : index, index, index
  }
  return %region: tensor<?x16x?xf32>
}

// CHECK-LABEL: func.func @empty_region
//  CHECK-SAME: (%[[TENSOR:.+]]: tensor<?x16x?xf32>, %[[DIM0:.+]]: index, %[[DIM2:.+]]: index)
//       CHECK:   %[[C16:.+]] = arith.constant 16 : index
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[DIMC0:.+]] = tensor.[[DIMC0]] %[[TENSOR]], %[[C0]] : tensor<?x16x?xf32>
//       CHECK:   %[[DDIM16:.+]] = flow.dispatch.dynamicize_dim 16 : index
//       CHECK:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[DIMC2:.+]] = tensor.[[DIMC0]] %[[TENSOR]], %[[C2]] : tensor<?x16x?xf32>
//       CHECK:   %[[SHAPE:.+]] = flow.dispatch.dynamicize_shape %[[TENSOR]] : tensor<?x16x?xf32> -> tensor<?x?x?xf32>{%[[DIMC0]], %[[DDIM16]], %[[DIMC2]]}
//       CHECK:   %[[REGION:.+]] = flow.dispatch.region[%[[DIM0]], %[[C16]], %[[DIM2]]] -> (tensor<?x?x?xf32>{%[[DIM0]], %[[DDIM16]], %[[DIM2]]}) {
//       CHECK:     flow.return %[[SHAPE]] : tensor<?x?x?xf32>
//       CHECK:   } count(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index) -> (index, index, index) {
//       CHECK:     %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %[[ARG0]], %[[ARG1]], %[[ARG2]]
//       CHECK:     flow.return %x, %y, %z : index, index, index
//       CHECK:   }
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[REGION]] : tensor<?x?x?xf32> to tensor<?x16x?xf32>
//       CHECK:   return %[[CAST]] : tensor<?x16x?xf32>

// -----

func.func @tensor_empty(%dim0: index, %dim2: index) -> tensor<?x16x?xf32> {
  %c16 = arith.constant 16 : index
  %region = flow.dispatch.region[%dim0, %c16, %dim2] -> (tensor<?x16x?xf32>{%dim0, %dim2}) {
    %empty = tensor.empty(%dim0, %dim2) : tensor<?x16x?xf32>
    flow.return %empty : tensor<?x16x?xf32>
  }
  return %region: tensor<?x16x?xf32>
}

// CHECK-LABEL: func.func @tensor_empty
//  CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM2:.+]]: index)
//       CHECK:   %[[DDIM16:.+]] = flow.dispatch.dynamicize_dim 16 : index
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DDIM16]], %[[DIM2]]) : tensor<?x?x?xf32>
//       CHECK:     flow.return %[[EMPTY]] : tensor<?x?x?xf32>

// -----

func.func @tensor_constant(%tensor: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %region = flow.dispatch.region[%c8, %c16] -> (tensor<8x16xf32>) {
    %cst = arith.constant dense<4.25> : tensor<8x16xf32>
    %empty = tensor.empty() : tensor<8x16xf32>
    %elementwise = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%tensor, %cst : tensor<8x16xf32>, tensor<8x16xf32>) outs(%empty : tensor<8x16xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %add = arith.addf %in0, %in1 : f32
      linalg.yield %add : f32
    } -> tensor<8x16xf32>
    flow.return %elementwise : tensor<8x16xf32>
  }
  return %region: tensor<8x16xf32>
}

// CHECK-LABEL: func.func @tensor_constant
//  CHECK-SAME: (%[[TENSOR:.+]]: tensor<8x16xf32>)
//       CHECK:   %[[DDIM8:.+]] = flow.dispatch.dynamicize_dim 8 : index
//       CHECK:   %[[DDIM16:.+]] = flow.dispatch.dynamicize_dim 16 : index
//       CHECK:   %[[SHAPE:.+]] = flow.dispatch.dynamicize_shape %[[TENSOR]] : tensor<8x16xf32> -> tensor<?x?xf32>{%[[DDIM8]], %[[DDIM16]]}
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[CST:.+]] = arith.constant dense<4.250000e+00> : tensor<8x16xf32>
//       CHECK:     %5 = linalg.generic
//  CHECK-SAME:       ins(%[[SHAPE]], %[[CST]] : tensor<?x?xf32>, tensor<8x16xf32>)

// -----

func.func @linalg_matmul_elementwise(%A: tensor<?x2048x?xf32>, %B: tensor<?x?x4096xf32>, %D: tensor<?x2048x4096xf32>, %dim: index) -> tensor<?x2048x4096xf32> {
  %c2048 = arith.constant 2048 : index
  %c4096 = arith.constant 4096 : index
  %f0 = arith.constant 0.0 : f32
  %region = flow.dispatch.region[%dim, %c2048, %c4096] -> (tensor<?x2048x4096xf32>{%dim}) {
    %empty = tensor.empty(%dim) : tensor<?x2048x4096xf32>
    %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<?x2048x4096xf32>) -> tensor<?x2048x4096xf32>
    %matmul = linalg.batch_matmul
      ins(%A, %B : tensor<?x2048x?xf32>, tensor<?x?x4096xf32>)
      outs(%fill : tensor<?x2048x4096xf32>) -> tensor<?x2048x4096xf32>
    %elementwise = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%matmul, %D : tensor<?x2048x4096xf32>, tensor<?x2048x4096xf32>) outs(%empty : tensor<?x2048x4096xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %add = arith.addf %in0, %in1 : f32
      linalg.yield %add : f32
    } -> tensor<?x2048x4096xf32>
    flow.return %elementwise : tensor<?x2048x4096xf32>
  }
  return %region: tensor<?x2048x4096xf32>
}

// CHECK-LABEL: func.func @linalg_matmul_elementwise
//  CHECK-SAME: (%[[A:.+]]: tensor<?x2048x?xf32>, %[[B:.+]]: tensor<?x?x4096xf32>, %[[D:.+]]: tensor<?x2048x4096xf32>, %[[DIM:.+]]: index)
//   CHECK-DAG:   %[[C2048:.+]] = arith.constant 2048 : index
//   CHECK-DAG:   %[[C4096:.+]] = arith.constant 4096 : index
//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[DIMA0:.+]] = tensor.dim %[[A]], %[[C0]] : tensor<?x2048x?xf32>
//       CHECK:   %[[DDIM2048:.+]] = flow.dispatch.dynamicize_dim 2048 : index
//       CHECK:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[DIMA2:.+]] = tensor.dim %[[A]], %[[C2]] : tensor<?x2048x?xf32>
//       CHECK:   %[[SHAPEA:.+]] = flow.dispatch.dynamicize_shape %[[A]] : tensor<?x2048x?xf32> -> tensor<?x?x?xf32>{%[[DIMA0]], %[[DDIM2048]], %[[DIMA2]]}
//       CHECK:   %[[DIMB0:.+]] = tensor.dim %[[B]], %[[C0]] : tensor<?x?x4096xf32>
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[DIMB1:.+]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?x4096xf32>
//       CHECK:   %[[DDIM4096:.+]] = flow.dispatch.dynamicize_dim 4096 : index
//       CHECK:   %[[SHAPEB:.+]] = flow.dispatch.dynamicize_shape %[[B]] : tensor<?x?x4096xf32> -> tensor<?x?x?xf32>{%[[DIMB0]], %[[DIMB1]], %[[DDIM4096]]}
//       CHECK:   %[[DIMD0:.+]] = tensor.dim %[[D]], %[[C0]] : tensor<?x2048x4096xf32>
//       CHECK:   %[[SHAPED:.+]] = flow.dispatch.dynamicize_shape %[[D]] : tensor<?x2048x4096xf32> -> tensor<?x?x?xf32>{%[[DIMD0]], %[[DDIM2048]], %[[DDIM4096]]}
//       CHECK:   %[[REGION:.+]] = flow.dispatch.region[%[[DIM]], %[[C2048]], %[[C4096]]] -> (tensor<?x?x?xf32>{%[[DIM]], %[[DDIM2048]], %[[DDIM4096]]})
//       CHECK:     %[[EMPTY:.+]] = tensor.empty(%[[DIM]], %[[DDIM2048]], %[[DDIM4096]]) : tensor<?x?x?xf32>
//       CHECK:     %[[FILL:.+]] = linalg.fill ins(%[[F0]] : f32) outs(%[[EMPTY]] : tensor<?x?x?xf32>)
//       CHECK:     %[[MATMUL:.+]] = linalg.batch_matmul
//  CHECK-SAME:       ins(%[[SHAPEA]], %[[SHAPEB]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
//  CHECK-SAME:       outs(%[[FILL]] : tensor<?x?x?xf32>)
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[MATMUL]], %[[SHAPED]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?x?xf32>)
//       CHECK:     flow.return %[[GENERIC]] : tensor<?x?x?xf32>
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[REGION]] : tensor<?x?x?xf32> to tensor<?x2048x4096xf32>
//       CHECK:   return %[[CAST]] : tensor<?x2048x4096xf32>

// -----

func.func @tensor_extract_slice(%tensor: tensor<4x32x384xf32>) -> tensor<8x384xf32> {
  %c8 = arith.constant 8 : index
  %c384 = arith.constant 384 : index
  %region = flow.dispatch.region[%c8, %c384] -> (tensor<8x384xf32>) {
    %extract = tensor.extract_slice %tensor[2, 4, 0] [1, 8, 384] [1, 1, 1] : tensor<4x32x384xf32> to tensor<8x384xf32>
    flow.return %extract : tensor<8x384xf32>
  }
  return %region: tensor<8x384xf32>
}

// CHECK-LABEL: func.func @tensor_extract_slice
//  CHECK-SAME: (%[[TENSOR:.+]]: tensor<4x32x384xf32>)
//   CHECK-DAG:   %[[DDIM4:.+]] = flow.dispatch.dynamicize_dim 4 : index
//   CHECK-DAG:   %[[DDIM32:.+]] = flow.dispatch.dynamicize_dim 32 : index
//   CHECK-DAG:   %[[DDIM384:.+]] = flow.dispatch.dynamicize_dim 384 : index
//       CHECK:   %[[SHAPE:.+]] = flow.dispatch.dynamicize_shape %[[TENSOR]] : tensor<4x32x384xf32> -> tensor<?x?x?xf32>{%[[DDIM4]], %[[DDIM32]], %[[DDIM384]]}
//   CHECK-DAG:   %[[DDIM0:.+]] = flow.dispatch.dynamicize_dim 0 : index
//   CHECK-DAG:   %[[DDIM2:.+]] = flow.dispatch.dynamicize_dim 2 : index
//   CHECK-DAG:   %[[DDIM8:.+]] = flow.dispatch.dynamicize_dim 8 : index
//       CHECK:   flow.dispatch.region[%c8, %c384] -> (tensor<?x?xf32>{%[[DDIM8]], %[[DDIM384]]})
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[SHAPE]][%[[DDIM2]], %[[DDIM4]], %[[DDIM0]]] [1, %[[DDIM8]], %[[DDIM384]]] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?xf32>
//       CHECK:     flow.return %[[EXTRACT]] : tensor<?x?xf32>

// -----

func.func @tensor_extract_slice(%tensor: tensor<?x?x?x?xf32>) -> tensor<8x1xf32> {
  %c8 = arith.constant 8 : index
  %c384 = arith.constant 384 : index
  %region = flow.dispatch.region[%c8, %c384] -> (tensor<8x1xf32>) {
    %extract = tensor.extract_slice %tensor[2, 4, 0, 16] [1, 8, 1, 1] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<8x1xf32>
    flow.return %extract : tensor<8x1xf32>
  }
  return %region: tensor<8x1xf32>
}

// CHECK-LABEL: func.func @tensor_extract_slice
//   CHECK-DAG:   %[[DDIM0:.+]] = flow.dispatch.dynamicize_dim 0 : index
//   CHECK-DAG:   %[[DDIM1:.+]] = flow.dispatch.dynamicize_dim 1 : index
//   CHECK-DAG:   %[[DDIM2:.+]] = flow.dispatch.dynamicize_dim 2 : index
//   CHECK-DAG:   %[[DDIM4:.+]] = flow.dispatch.dynamicize_dim 4 : index
//   CHECK-DAG:   %[[DDIM8:.+]] = flow.dispatch.dynamicize_dim 8 : index
//   CHECK-DAG:   %[[DDIM16:.+]] = flow.dispatch.dynamicize_dim 16 : index
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %{{.+}}[%[[DDIM2]], %[[DDIM4]], %[[DDIM0]], %[[DDIM16]]] [1, %[[DDIM8]], 1, %[[DDIM1]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?xf32>
//       CHECK:     flow.return %[[EXTRACT]] : tensor<?x?xf32>

// -----

func.func @tensor_insert_slice(%partial: tensor<384x128xf32>, %full: tensor<1x384x384xf32>) -> tensor<1x384x384xf32> {
  %c384 = arith.constant 384 : index
  %c128 = arith.constant 128 : index
  %region = flow.dispatch.region[%c384, %c128] -> (tensor<1x384x384xf32>) {
    %insert = tensor.insert_slice %partial into %full[0, 0, 0] [1, 384, 128] [1, 1, 1] : tensor<384x128xf32> into tensor<1x384x384xf32>
    flow.return %insert : tensor<1x384x384xf32>
  }
  return %region: tensor<1x384x384xf32>
}

// CHECK-LABEL: func.func @tensor_insert_slice
//  CHECK-SAME: (%[[PARTIAL:.+]]: tensor<384x128xf32>, %[[FULL:.+]]: tensor<1x384x384xf32>)
//   CHECK-DAG:   %[[DDIM384:.+]] = flow.dispatch.dynamicize_dim 384 : index
//   CHECK-DAG:   %[[DDIM128:.+]] = flow.dispatch.dynamicize_dim 128 : index
//       CHECK:   %[[SHAPEP:.+]] = flow.dispatch.dynamicize_shape %[[PARTIAL]] : tensor<384x128xf32> -> tensor<?x?xf32>{%[[DDIM384]], %[[DDIM128]]}
//       CHECK:   %[[DDIM1:.+]] = flow.dispatch.dynamicize_dim 1 : index
//       CHECK:   %[[SHAPEF:.+]] = flow.dispatch.dynamicize_shape %[[FULL]] : tensor<1x384x384xf32> -> tensor<?x?x?xf32>{%[[DDIM1]], %0, %0}
//       CHECK:   %[[DDIM0:.+]] = flow.dispatch.dynamicize_dim 0 : index
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[INSERT:.+]] = tensor.insert_slice %[[SHAPEP]] into %[[SHAPEF]][%[[DDIM0]], %[[DDIM0]], %[[DDIM0]]] [1, %[[DDIM384]], %[[DDIM128]]] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
//       CHECK:     flow.return %[[INSERT]] : tensor<?x?x?xf32>
