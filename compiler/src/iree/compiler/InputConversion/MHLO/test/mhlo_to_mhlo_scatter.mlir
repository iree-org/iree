// RUN: iree-opt --split-input-file --verify-diagnostics --iree-mhlo-to-mhlo-preprocessing %s | FileCheck %s

func.func @scatter_implicit_batch(%arg0: tensor<5x6x7xi32>, %arg1: tensor<2xi32>, %arg2: tensor<7xi32>) -> tensor<5x6x7xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true} : (tensor<5x6x7xi32>, tensor<2xi32>, tensor<7xi32>) -> tensor<5x6x7xi32>
  return %0 : tensor<5x6x7xi32>
}

// CHECK-LABEL: func.func @scatter_implicit_batch
// CHECK-DAG: %[[RE_I:.+]] = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] : tensor<2xi32> into tensor<1x2xi32>
// CHECK-DAG: %[[RE_U:.+]] = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] : tensor<7xi32> into tensor<1x7xi32>
// CHECK:     %[[SCATTER:.+]] = "mhlo.scatter"(%{{.*}}, %[[RE_I]], %[[RE_U]])
// CHECK:       mhlo.return %{{.*}}
// CHECK:            update_window_dims = [1],
// CHECK-SAME:       inserted_window_dims = [0, 1]
// CHECK-SAME:       scatter_dims_to_operand_dims = [0, 1]

// -----

func.func @scatter_implicit_indices(%arg0: tensor<17x11xf32>,
  %arg1: tensor<7xi32>, %arg2: tensor<7x11xf32>) -> tensor<17x11xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {indices_are_sorted = false,
      scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
      unique_indices = false
      } : (tensor<17x11xf32>, tensor<7xi32>, tensor<7x11xf32>) -> tensor<17x11xf32>
  return %0 : tensor<17x11xf32>
}

// CHECK-LABEL: func.func @scatter_implicit_indices
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %arg1 {{\[\[}}0, 1]] : tensor<7xi32> into tensor<7x1xi32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%arg0, %[[EXPAND]], %arg2) ({
// CHECK-NEXT: ^bb0(%[[A0:.+]]: tensor<f32>, %[[A1:.+]]: tensor<f32>):
// CHECK-NEXT:   %[[ADD:.+]] = mhlo.add %[[A0]], %[[A1]] : tensor<f32>
// CHECK-NEXT:   mhlo.return %[[ADD]]
// CHECK-NEXT: })
// CHECK-SAME: indices_are_sorted = false,
// CHECK-SAME: scatter_dimension_numbers = #mhlo.scatter<
// CHECK-SAME:   update_window_dims = [1],
// CHECK-SAME:   inserted_window_dims = [0],
// CHECK-SAME:   scatter_dims_to_operand_dims = [0],
// CHECK-SAME:   index_vector_dim = 1>,
// CHECK-SAME:   unique_indices = false

// -----

func.func @scatter_collapse_batch(%arg0: tensor<1x24x512xi32>,
    %arg1: tensor<2x3x2xi32>, %arg2: tensor<2x3x512xi32>) -> tensor<1x24x512xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {indices_are_sorted = false,
      scatter_dimension_numbers = #mhlo.scatter<
        update_window_dims = [2],
        inserted_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 2,
      >,
      unique_indices = true
  } : (tensor<1x24x512xi32>, tensor<2x3x2xi32>, tensor<2x3x512xi32>) -> tensor<1x24x512xi32>
  return %0 : tensor<1x24x512xi32>
}

// CHECK-LABEL: func.func @scatter_collapse_batch
// CHECK: %[[COLLAPSE0:.+]] = tensor.collapse_shape %arg1 {{\[\[}}0, 1], [2]] : tensor<2x3x2xi32> into tensor<6x2xi32>
// CHECK: %[[COLLAPSE1:.+]] = tensor.collapse_shape %arg2 {{\[\[}}0, 1], [2]] : tensor<2x3x512xi32> into tensor<6x512xi32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%arg0, %[[COLLAPSE0]], %[[COLLAPSE1]])
// CHECK: ^bb0(%[[ARG0:.+]]: tensor<i32>, %[[ARG1:.+]]: tensor<i32>):
// CHECK:   mhlo.return %[[ARG1]]
// CHECK: }) {
// CHECK: indices_are_sorted = false,
// CHECK-SAME: scatter_dimension_numbers = #mhlo.scatter<
// CHECK-SAME: update_window_dims = [1]
// CHECK-SAME: inserted_window_dims = [0, 1]
// CHECK-SAME: scatter_dims_to_operand_dims = [0, 1]
// CHECK-SAME: index_vector_dim = 1>
// CHECK-SAME: unique_indices = true
// CHECK: return %[[SCATTER]]

// -----

func.func @scatter_materialize_index_update(%arg0: tensor<5x1x1xi32>, %arg1: tensor<1x2xi32>, %arg2: tensor<1x4xi32>) -> tensor<5x1x1xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1],
                                              inserted_window_dims = [1, 2],
                                              scatter_dims_to_operand_dims = [0, 1],
                                              index_vector_dim = 1>,
    unique_indices = true} : (tensor<5x1x1xi32>, tensor<1x2xi32>, tensor<1x4xi32>) -> tensor<5x1x1xi32>
  return %0 : tensor<5x1x1xi32>
}

// CHECK-LABEL: @scatter_materialize_index_update
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %arg2 {{\[\[}}0], [1, 2, 3]] : tensor<1x4xi32> into tensor<1x4x1x1xi32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%arg0, %arg1, %[[EXPAND]])
// CHECK:                   indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<
// CHECK-SAME:                update_window_dims = [1, 2, 3]
// CHECK-SAME:                scatter_dims_to_operand_dims = [0, 1]
// CHECK-SAME:                index_vector_dim = 1>, unique_indices = true

// -----

func.func @scatter_materialize_one_dim(%arg0: tensor<5x1x1xi32>, %arg1: tensor<1x2xi32>, %arg2: tensor<1xi32>) -> tensor<5x1x1xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [],
                                              inserted_window_dims = [0, 1, 2],
                                              scatter_dims_to_operand_dims = [0, 1],
                                              index_vector_dim = 1>,
    unique_indices = true} : (tensor<5x1x1xi32>, tensor<1x2xi32>, tensor<1xi32>) -> tensor<5x1x1xi32>
  return %0 : tensor<5x1x1xi32>
}

// CHECK-LABEL: @scatter_materialize_one_dim
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %arg2 {{\[\[}}0, 1]] : tensor<1xi32> into tensor<1x1xi32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%arg0, %arg1, %[[EXPAND]])
// CHECK:                   indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<
// CHECK-SAME:                 update_window_dims = [1]
// CHECK-SAME:                 inserted_window_dims = [0, 1]
// CHECK-SAME:                 scatter_dims_to_operand_dims = [0, 1]
// CHECK-SAME:                 index_vector_dim = 1>, unique_indices = true

// -----

func.func @scatter_materialize_two_dims(%arg0: tensor<5x1x1xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1xi32>) -> tensor<5x1x1xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [],
                                              inserted_window_dims = [0, 1, 2],
                                              scatter_dims_to_operand_dims = [0],
                                              index_vector_dim = 1>,
    unique_indices = true} : (tensor<5x1x1xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<5x1x1xi32>
  return %0 : tensor<5x1x1xi32>
}

// CHECK-LABEL: @scatter_materialize_two_dims
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %arg2 {{\[\[}}0, 1, 2]] : tensor<1xi32> into tensor<1x1x1xi32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%arg0, %arg1, %[[EXPAND]])
// CHECK:                   indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<
// CHECK-SAME:                 update_window_dims = [1, 2]
// CHECK-SAME:                 inserted_window_dims = [0]
// CHECK-SAME:                 scatter_dims_to_operand_dims = [0]
// CHECK-SAME:                 index_vector_dim = 1>, unique_indices = true

// -----

func.func @scatter_materialize_comprehensive(%arg0: tensor<5x4x1xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x4xi32>) -> tensor<5x4x1xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1],
                                              inserted_window_dims = [0, 2],
                                              scatter_dims_to_operand_dims = [0],
                                              index_vector_dim = 1>,
    unique_indices = true} : (tensor<5x4x1xi32>, tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<5x4x1xi32>
  return %0 : tensor<5x4x1xi32>
}

// CHECK-LABEL: @scatter_materialize_comprehensive
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %arg2 {{\[\[}}0], [1, 2]] : tensor<1x4xi32> into tensor<1x4x1xi32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%arg0, %arg1, %[[EXPAND]])
// CHECK:                   indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<
// CHECK-SAME:                 update_window_dims = [1, 2]
// CHECK-SAME:                 inserted_window_dims = [0]
// CHECK-SAME:                 scatter_dims_to_operand_dims = [0]
// CHECK-SAME:                 index_vector_dim = 1>, unique_indices = true

// -----

func.func @scatter_operand_map(%arg0: tensor<5x4x1xi32>, %arg1: tensor<1x2xi32>, %arg2: tensor<1xi32>) -> tensor<5x4x1xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [],
                                              inserted_window_dims = [0, 1, 2],
                                              scatter_dims_to_operand_dims = [0, 2],
                                              index_vector_dim = 1>,
    unique_indices = true} : (tensor<5x4x1xi32>, tensor<1x2xi32>, tensor<1xi32>) -> tensor<5x4x1xi32>
  return %0 : tensor<5x4x1xi32>
}

// CHECK-LABEL: @scatter_operand_map
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %arg2 {{\[\[}}0, 1, 2]] : tensor<1xi32> into tensor<1x1x1xi32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%arg0, %arg1, %[[EXPAND]])
// CHECK:                   indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<
// CHECK-SAME:                 update_window_dims = [1, 2],
// CHECK-SAME:                 inserted_window_dims = [0],
// CHECK-SAME:                 scatter_dims_to_operand_dims = [0, 2],
// CHECK-SAME:                 index_vector_dim = 1>, unique_indices = true

// -----

func.func @scatter_update_transpose(%a: tensor<16x17x8x384xf32>, %b: tensor<15x1xi32>, %c: tensor<16x17x15x384xf32>) -> tensor<16x17x8x384xf32>
{
  %out = "mhlo.scatter"(%a, %b, %c) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %add = mhlo.add %arg0, %arg1 : tensor<f32>
      mhlo.return %add : tensor<f32>
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 1, 3],
        inserted_window_dims = [2],
        scatter_dims_to_operand_dims = [2],
        index_vector_dim = 1>,
        unique_indices = false} : (tensor<16x17x8x384xf32>, tensor<15x1xi32>, tensor<16x17x15x384xf32>) -> tensor<16x17x8x384xf32>
  return %out : tensor<16x17x8x384xf32>
}

// CHECK-LABEL: @scatter_update_transpose
// CHECK-SAME: %[[ARG0:.+]]: tensor<16x17x8x384xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<15x1xi32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<16x17x15x384xf32>
// CHECK:   %[[TRANSPOSE:.+]] = "mhlo.transpose"(%[[ARG2]]) {permutation = dense<[2, 0, 1, 3]> : tensor<4xi64>}
// CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[TRANSPOSE]]
// CHECK-NEXT{literal}: [[0], [1], [2, 3], [4]] : tensor<15x16x17x384xf32> into tensor<15x16x17x1x384xf32>
// CHECK:   %[[SCATTER:.+]] = "mhlo.scatter"(%[[ARG0]], %[[ARG1]], %[[EXPANDED]]) ({
// CHECK:   ^bb0(%[[ARG3:.+]]: tensor<f32>, %[[ARG4:.+]]: tensor<f32>):
// CHECK:     %[[ADD:.+]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:     mhlo.return %[[ADD]] : tensor<f32>
// CHECK:   }) 
// CHECK-SAME: indices_are_sorted = false
// CHECK-SAME: scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>
// CHECK-SAME: unique_indices = false
// CHECK:   return %[[SCATTER]]

// -----

func.func @scatter_transpose_indices(%arg0: tensor<1x64x32x640xf32>, %arg1: tensor<1x44xi32>, %arg2: tensor<44x1x64x640xf32>) -> tensor<1x64x32x640xf32> {
  %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x44xi32>) -> tensor<44x1xi32>
  %expanded = tensor.expand_shape %arg2 [[0], [1], [2, 3], [4]] : tensor<44x1x64x640xf32> into tensor<44x1x64x1x640xf32>
  %1 = "mhlo.scatter"(%arg0, %0, %expanded) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = mhlo.add %arg3, %arg4 : tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false} : (tensor<1x64x32x640xf32>, tensor<44x1xi32>, tensor<44x1x64x1x640xf32>) -> tensor<1x64x32x640xf32>
  return %1 : tensor<1x64x32x640xf32>
}

// CHECK-LABEL: @scatter_transpose_indices
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x64x32x640xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x44xi32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<44x1x64x640xf32>
// CHECK: %[[TRANSPOSE:.+]] = "mhlo.transpose"(%[[ARG1]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
// CHECK: %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG2]]
// CHECK-SAME{literal}: [[0], [1], [2, 3], [4]] : tensor<44x1x64x640xf32> into tensor<44x1x64x1x640xf32>
// CHECK: %[[SCATTER:.+]] = "mhlo.scatter"(%[[ARG0]], %[[TRANSPOSE]], %[[EXPANDED]])
// CHECK: ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK:   %2 = mhlo.add %arg3, %arg4 : tensor<f32>
// CHECK:   mhlo.return %2 : tensor<f32>
// CHECK: indices_are_sorted = false
// CHECK-SAME: scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>
// CHECK-SAME: unique_indices = false
// CHECK: return %[[SCATTER]] : tensor<1x64x32x640xf32>

// -----

func.func @scatter_i64_indices(%arg0: tensor<5x6x7xi32>, %arg1: tensor<1x2xi64>, %arg2: tensor<1x7xi32>) -> tensor<5x6x7xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    mhlo.return %arg4 : tensor<i32>
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi32>, tensor<1x2xi64>, tensor<1x7xi32>) -> tensor<5x6x7xi32>
  return %0 : tensor<5x6x7xi32>
}

// CHECK-LABEL: func.func @scatter_i64_indices
// CHECK-SAME: %[[ARG0:.+]]: tensor<5x6x7xi32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x2xi64>
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x7xi32>
// CHECK-DAG: %[[CONVERT:.+]] = mhlo.convert %[[ARG1]] : (tensor<1x2xi64>) -> tensor<1x2xi32>
// CHECK:     %[[SCATTER:.+]] = "mhlo.scatter"(%[[ARG0]], %[[CONVERT]], %[[ARG2]])
// CHECK:       mhlo.return %{{.*}}
// CHECK:            update_window_dims = [1],
// CHECK-SAME:       inserted_window_dims = [0, 1]
// CHECK-SAME:       scatter_dims_to_operand_dims = [0, 1]
