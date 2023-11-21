// RUN: iree-opt --iree-stablehlo-limit-scatter-bounds --cse %s | FileCheck %s

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: @limit_scatter
func.func @limit_scatter(%arg0 : tensor<10x3x8xi32>, %arg1 : tensor<4x2xi32>, %arg2 : tensor<4x2x8xi32>) -> tensor<10x3x8xi32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[IND0:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK-DAG: %[[IND1:.+]] = tensor.dim %arg0, %[[C1]]
  // CHECK-DAG: %[[IND2:.+]] = tensor.dim %arg0, %[[C2]]
  // CHECK-DAG: %[[UPD1:.+]] = tensor.dim %arg2, %[[C1]]
  // CHECK-DAG: %[[UPD2:.+]] = tensor.dim %arg2, %[[C2]]
  // CHECK-DAG: %[[SUB0:.+]] = arith.subi %[[IND0]], %[[UPD1]]
  // CHECK-DAG: %[[SUB1:.+]] = arith.subi %[[IND1]], %[[C1]]
  // CHECK-DAG: %[[CAST0:.+]] = arith.index_cast %[[SUB1]]
  // CHECK-DAG: %[[CAST1:.+]] = arith.index_cast %[[SUB0]]
  // CHECK: %[[LIMITS:.+]] = tensor.from_elements %[[CAST0]], %[[CAST1]]


  // We check the indices agains the bounds and save out whether a bound was
  // violated and force it within bounds if it was:
  // CHECK: %[[EMPTY_IND:.+]] = tensor.empty() : tensor<4x2xi32>
  // CHECK: %[[EMPTY_BOUNDS:.+]] = tensor.empty() : tensor<4x2xi1>

  // CHECK: %[[BOUND:.+]]:2 = linalg.generic 
  // CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP0]], #[[MAP0]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME: ins(%arg1, %[[LIMITS]] : tensor<4x2xi32>, tensor<2xi32>)
  // CHECK-SAME: outs(%[[EMPTY_IND]], %[[EMPTY_BOUNDS]] : tensor<4x2xi32>, tensor<4x2xi1>)
  // CHECK: ^bb0(%[[IN0:.+]]: i32, %[[IN1:.+]]: i32, %[[OUT0:.+]]: i32, %[[OUT1:.+]]: i1):
  // CHECK:   %[[I0:.+]] = arith.constant 0 : i32
  // CHECK:   %[[GE:.+]] = arith.cmpi sge, %[[IN0]], %[[I0]]
  // CHECK:   %[[LE:.+]] = arith.cmpi sle, %[[IN0]], %[[IN1]]
  // CHECK:   %[[AND:.+]] = arith.andi %[[GE]], %[[LE]]
  // CHECK:   %[[SEL:.+]] = arith.select %[[AND]], %[[IN0]], %[[I0]] : i32
  // CHECK:   linalg.yield %[[SEL]], %[[AND]] : i32, i1

  // CHECK: %[[TRUE:.+]] = arith.constant true
  // CHECK: %[[SPLAT:.+]] = tensor.splat %[[TRUE]]
  // CHECK: %[[REDUCE:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP2]]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME: ins(%[[BOUND]]#1 : tensor<4x2xi1>)
  // CHECK-SAME: outs(%[[SPLAT]] : tensor<4xi1>)
  // CHECK: ^bb0(%[[IN:.+]]: i1, %[[OUT:.+]]: i1):
  // CHECK:   %[[AND:.+]] = arith.andi %[[IN]], %[[OUT]] : i1
  // CHECK:   linalg.yield %[[AND]]

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x2x8xi1>
  // CHECK: %[[BROADCAST:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map3, #map4]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[REDUCE]] : tensor<4xi1>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<4x2x8xi1>)
  // CHECK: ^bb0(%[[IN:.+]]: i1, %[[OUT:.+]]: i1):
  // CHECK:   linalg.yield %[[IN]]

  // CHECK:  %[[SCATTER_BOUND:.+]] = tensor.empty() : tensor<10x3x8xi1>
  // CHECK: %[[SCATTER:.+]]:2 = "stablehlo.scatter"(%arg0, %[[SCATTER_BOUND]], %[[BOUND]]#0, %arg2, %[[BROADCAST]])

  // CHECK: ^bb0(%[[A0:.+]]: tensor<i32>, %[[A1:.+]]: tensor<i1>, %[[A2:.+]]: tensor<i32>, %[[A3:.+]]: tensor<i1>):
  // CHECK:   %[[SEL:.+]] = stablehlo.select %[[A3]], %[[A2]], %[[A0]]
  // CHECK:   stablehlo.return %[[SEL]], %[[A3]]
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [1],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<10x3x8xi32>, tensor<4x2xi32>, tensor<4x2x8xi32>) -> tensor<10x3x8xi32>

  // CHECK: return %[[SCATTER]]#0
  return %0 : tensor<10x3x8xi32>
}
