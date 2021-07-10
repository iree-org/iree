// RUN: iree-opt -split-input-file -iree-mhlo-to-linalg-ext %s | IreeFileCheck %s

func @sort_1d(%arg0: tensor<128xi32>) -> (tensor<128xi32>) {
  %0 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<128xi32>) -> (tensor<128xi32>)
  return %0 : tensor<128xi32>
}
// CHECK-LABEL: func @sort_1d
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]]) : (tensor<128xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG1:.+]]: !flow.dispatch.tensor<readwrite:128xi32>
// CHECK:           %[[IN:.+]] = flow.dispatch.tensor.load %[[ARG1]]
// CHECK:           %[[SORT:.+]] = linalg_ext.sort
// CHECK-SAME:        dimension(0)
// CHECK-SAME:        outs(%[[IN]] : tensor<128xi32>)
// CHECK:          ^bb0(%[[ARG2:.+]]: i32, %[[ARG3:.+]]: i32)
// CHECK:            %[[CMP:.+]] = cmpi sgt, %[[ARG2]], %[[ARG3]]
// CHECK:            linalg_ext.yield %[[CMP]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]], %[[ARG1]]
// CHECK:        return %[[RES]]

// -----

func @sort_2d(%arg0: tensor<16x32xi32>) -> (tensor<16x32xi32>) {
  %0 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<16x32xi32>) -> (tensor<16x32xi32>)
  return %0 : tensor<16x32xi32>
}
// CHECK-LABEL: func @sort_2d
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]]) : (tensor<16x32xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG1:.+]]: !flow.dispatch.tensor<readwrite:16x32xi32>
// CHECK:           %[[IN:.+]] = flow.dispatch.tensor.load %[[ARG1]]
// CHECK:           %[[SORT:.+]] = linalg_ext.sort
// CHECK-SAME:        dimension(0)
// CHECK-SAME:        outs(%[[IN]] : tensor<16x32xi32>)
// CHECK:          ^bb0(%[[ARG2:.+]]: i32, %[[ARG3:.+]]: i32)
// CHECK:            %[[CMP:.+]] = cmpi sgt, %[[ARG2]], %[[ARG3]]
// CHECK:            linalg_ext.yield %[[CMP]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]], %[[ARG1]]
// CHECK:        return %[[RES]]

// -----

func @topk(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> (tensor<128xi32>) {
  %0:2 = "mhlo.sort"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):  // no predecessors
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<128xi32>, tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  return %0#0 : tensor<128xi32>
}
// CHECK-LABEL: func @topk
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]]:2 = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]])
// CHECK-SAME:    : (tensor<128xi32>, tensor<128xi32>) -> (%[[ARG0]], %[[ARG1]])
// CHECK:           %[[ARG2:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:128xi32>
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:128xi32>
// CHECK:           %[[IN1:.+]] = flow.dispatch.tensor.load %[[ARG2]]
// CHECK:           %[[IN2:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[SORT:.+]]:2 = linalg_ext.sort
// CHECK-SAME:        dimension(0)
// CHECK-SAME:        outs(%[[IN1]], %[[IN2]] : tensor<128xi32>, tensor<128xi32>)
// CHECK:          ^bb0(%[[ARG4:.+]]: i32, %[[ARG5:.+]]: i32, %{{.*}}: i32, %{{.*}}: i32)
// CHECK:            %[[CMP:.+]] = cmpi sgt, %[[ARG4]], %[[ARG5]]
// CHECK:            linalg_ext.yield %[[CMP]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]]#0, %[[ARG2]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]]#1, %[[ARG3]]
// CHECK:        return %[[RES]]#0

// -----

func @scatter_update_scalar_1D(%arg0: tensor<8xi32>, %arg1: tensor<4x1xi32>,
    %arg2: tensor<4xi32>) -> tensor<8xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<> : tensor<0xi64>
    },
    unique_indices = false
  } : (tensor<8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: func @scatter_update_scalar_1D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-SAME:    : (tensor<8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:8xi32>
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:4x1xi32>
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:4xi32>
// CHECK:           %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG4]]
// CHECK:           %[[UPDATES:.+]] = flow.dispatch.tensor.load %[[ARG5]]
// CHECK:           %[[SCATTER:.+]] = linalg_ext.scatter
// CHECK-SAME:        ins(%[[UPDATES]], %[[INDICES]] : tensor<4xi32>, tensor<4x1xi32>)
// CHECK-SAME:       outs(%[[ORIGINAL]] : tensor<8xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):  // no predecessors
// CEECK:             linalg.yield %[[V1]]
// CHECK:           flow.dispatch.tensor.store %[[SCATTER]], %[[ARG3]]
// CHECK:        return %[[RES]]

// -----

func @scatter_update_scalar_2D(%arg0: tensor<4x3xi32>, %arg1: tensor<3x2xi32>,
    %arg2: tensor<3xi32>) -> tensor<4x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {indices_are_sorted = false,
      scatter_dimension_numbers = {
        index_vector_dim = 1 : i64,
        inserted_window_dims = dense<[0, 1]> : tensor<2xi64>,
        scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
        update_window_dims = dense<> : tensor<0xi64>
      },
      unique_indices = false
  } : (tensor<4x3xi32>, tensor<3x2xi32>, tensor<3xi32>) -> tensor<4x3xi32>
  return %0 : tensor<4x3xi32>
}
// CHECK-LABEL: func @scatter_update_scalar_2D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-SAME:    : (tensor<4x3xi32>, tensor<3x2xi32>, tensor<3xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:4x3xi32>
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:3x2xi32>
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:3xi32>
// CHECK:           %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG4]]
// CHECK:           %[[UPDATES:.+]] = flow.dispatch.tensor.load %[[ARG5]]
// CHECK:           %[[SCATTER:.+]] = linalg_ext.scatter
// CHECK-SAME:        ins(%[[UPDATES]], %[[INDICES]] : tensor<3xi32>, tensor<3x2xi32>)
// CHECK-SAME:       outs(%[[ORIGINAL]] : tensor<4x3xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):  // no predecessors
// CEECK:             linalg.yield %[[V1]]
// CHECK:           flow.dispatch.tensor.store %[[SCATTER]], %[[ARG3]]
// CHECK:        return %[[RES]]

// -----

func @scatter_update_slice_2D(%arg0: tensor<6x3xi32>, %arg1: tensor<2x1xi32>,
    %arg2: tensor<2x3xi32>) -> tensor<6x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<1> : tensor<1xi64>
    },
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  return %0 : tensor<6x3xi32>
}
// CHECK-LABEL: func @scatter_update_slice_2D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-SAME:    : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:6x3xi32>
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:2x1xi32>
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:2x3xi32>
// CHECK:           %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG4]]
// CHECK:           %[[UPDATES:.+]] = flow.dispatch.tensor.load %[[ARG5]]
// CHECK:           %[[SCATTER:.+]] = linalg_ext.scatter
// CHECK-SAME:        ins(%[[UPDATES]], %[[INDICES]] : tensor<2x3xi32>, tensor<2x1xi32>)
// CHECK-SAME:       outs(%[[ORIGINAL]] : tensor<6x3xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):  // no predecessors
// CEECK:             linalg.yield %[[V1]]
// CHECK:           flow.dispatch.tensor.store %[[SCATTER]], %[[ARG3]]
// CHECK:        return %[[RES]]

// -----

func @scatter_add_slice_2D(%arg0: tensor<6x3xi32>, %arg1: tensor<2x1xi32>,
    %arg2: tensor<2x3xi32>) -> tensor<6x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<1> : tensor<1xi64>
    },
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  return %0 : tensor<6x3xi32>
}
// CHECK-LABEL: func @scatter_add_slice_2D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-SAME:    : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:6x3xi32>
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:2x1xi32>
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:2x3xi32>
// CHECK:           %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG4]]
// CHECK:           %[[UPDATES:.+]] = flow.dispatch.tensor.load %[[ARG5]]
// CHECK:           %[[SCATTER:.+]] = linalg_ext.scatter
// CHECK-SAME:        ins(%[[UPDATES]], %[[INDICES]] : tensor<2x3xi32>, tensor<2x1xi32>)
// CHECK-SAME:       outs(%[[ORIGINAL]] : tensor<6x3xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):  // no predecessors
//
//                   The order is reverse.
// CHECK:             %[[V3:.+]] = addi %[[V2]], %[[V1]]
// CEECK:             linalg.yield %[[V3]]
// CHECK:           flow.dispatch.tensor.store %[[SCATTER]], %[[ARG3]]
// CHECK:        return %[[RES]]

// -----

func @scatter_update_batch_scalar_1D(%arg0: tensor<8xi32>,
    %arg1: tensor<3x4x1xi32>, %arg2: tensor<3x4xi32>) -> tensor<8xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 2 : i64,
      inserted_window_dims = dense<0> : tensor<i64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<> : tensor<0xi64>
    },
    unique_indices = false
  } : (tensor<8xi32>, tensor<3x4x1xi32>, tensor<3x4xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: func @scatter_update_batch_scalar_1D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-SAME:    : (tensor<8xi32>, tensor<3x4x1xi32>, tensor<3x4xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:8xi32>
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:3x4x1xi32>
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:3x4xi32>
// CHECK:           %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG4]]
// CHECK:           %[[UPDATES:.+]] = flow.dispatch.tensor.load %[[ARG5]]
// CHECK:           %[[COLLAPSED_INDICES:.+]] = linalg.tensor_collapse_shape
// CHECK-SAME:        %[[INDICES]] {{\[}}[0, 1], [2]] : tensor<3x4x1xi32> into tensor<12x1xi32>
// CHECK:           %[[COLLAPSED_UPDATES:.+]] = linalg.tensor_collapse_shape
// CHECK-SAME:        %[[UPDATES]] {{\[}}[0, 1]] : tensor<3x4xi32> into tensor<12xi32>
// CHECK:           %[[SCATTER:.+]] = linalg_ext.scatter
// CHECK-SAME:        ins(%[[COLLAPSED_UPDATES]], %[[COLLAPSED_INDICES]] : tensor<12xi32>, tensor<12x1xi32>)
// CHECK-SAME:       outs(%[[ORIGINAL]] : tensor<8xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):  // no predecessors
// CEECK:             linalg.yield %[[V1]]
// CHECK:           flow.dispatch.tensor.store %[[SCATTER]], %[[ARG3]]
// CHECK:        return %[[RES]]

// -----

func @scatter_update_batch_slice_3D_dynamic(%arg0: tensor<1x24x512xi32>,
    %arg1: tensor<?x3x2xi32>, %arg2: tensor<?x3x512xi32>) -> tensor<1x24x512xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {indices_are_sorted = false,
      scatter_dimension_numbers = {
        index_vector_dim = 2 : i64,
        inserted_window_dims = dense<[0, 1]> : tensor<2xi64>,
        scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
        update_window_dims = dense<2> : tensor<1xi64>
      },
      unique_indices = false
  } : (tensor<1x24x512xi32>, tensor<?x3x2xi32>, tensor<?x3x512xi32>) -> tensor<1x24x512xi32>
  return %0 : tensor<1x24x512xi32>
}
// CHECK-LABEL: func @scatter_update_batch_slice_3D_dynamic
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[DIM1:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x3x2xi32>
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[DIM2:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x3x512xi32>
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-SAME:    : (tensor<1x24x512xi32>, tensor<?x3x2xi32>{%[[DIM1]]}, tensor<?x3x512xi32>{%[[DIM2]]}) -> %[[ARG0]]
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:1x24x512xi32>
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:?x3x2xi32>
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:?x3x512xi32>
// CHECK:           %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG4]]
// CHECK:           %[[UPDATES:.+]] = flow.dispatch.tensor.load %[[ARG5]]
// CHECK:           %[[COLLAPSED_INDICES:.+]] = linalg.tensor_collapse_shape
// CHECK-SAME:        %[[INDICES]] {{\[}}[0, 1], [2]] : tensor<?x3x2xi32> into tensor<?x2xi32>
// CHECK:           %[[COLLAPSED_UPDATES:.+]] = linalg.tensor_collapse_shape
// CHECK-SAME:        %[[UPDATES]] {{\[}}[0, 1], [2]] : tensor<?x3x512xi32> into tensor<?x512xi32>
// CHECK:           %[[SCATTER:.+]] = linalg_ext.scatter
// CHECK-SAME:        ins(%[[COLLAPSED_UPDATES]], %[[COLLAPSED_INDICES]] : tensor<?x512xi32>, tensor<?x2xi32>)
// CHECK-SAME:       outs(%[[ORIGINAL]] : tensor<1x24x512xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):  // no predecessors
// CEECK:             linalg.yield %[[V1]]
// CHECK:           flow.dispatch.tensor.store %[[SCATTER]], %[[ARG3]]
// CHECK:        return %[[RES]]
