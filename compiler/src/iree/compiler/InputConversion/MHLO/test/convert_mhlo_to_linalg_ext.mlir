// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-ext %s | FileCheck %s
// Also ensure that full lowering to linalg doesn't error.
// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-ext --iree-mhlo-to-linalg-on-tensors --reconcile-unrealized-casts %s

func.func @sort_1d(%arg0: tensor<128xi32>) -> (tensor<128xi32>) {
  %0 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<128xi32>) -> (tensor<128xi32>)
  return %0 : tensor<128xi32>
}
// CHECK-LABEL: func.func @sort_1d(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<128xi32>)
// CHECK:           ^bb0(%[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
// CHECK:             %[[CMP:.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG2]]
// CHECK:             iree_linalg_ext.yield %[[CMP]]
// CHECK:         return %[[SORT]]

// -----

func.func @sort_1d_ui(%arg0: tensor<128xui32>) -> (tensor<128xui32>) {
  %0 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<ui32>, %arg3: tensor<ui32>):  // no predecessors
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<128xui32>) -> (tensor<128xui32>)
  return %0 : tensor<128xui32>
}
// CHECK-LABEL: func.func @sort_1d_ui(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[CAST:.+]] = tensor.bitcast %[[ARG0]] : tensor<128xui32> to tensor<128xi32>
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs(%[[CAST]] : tensor<128xi32>)
// CHECK:           ^bb0(%[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
// CHECK:             %[[CMP:.+]] = arith.cmpi ugt, %[[ARG1]], %[[ARG2]]
// CHECK:             iree_linalg_ext.yield %[[CMP]]
// CHECK:         %[[RESULT:.+]] = tensor.bitcast %[[SORT]] : tensor<128xi32> to tensor<128xui32>
// CHECK:         return %[[RESULT]]

// -----

func.func @sort_cst_capture(%arg0: tensor<1x10xi32>) -> tensor<1x10xi32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<i32>, %arg3: tensor<i32>):
    %2 = "mhlo.compare"(%arg1, %0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x10xi32>) -> tensor<1x10xi32>
  return %1 : tensor<1x10xi32>
}

// CHECK-LABEL: func.func @sort_cst_capture(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[SCALAR:.+]] = arith.constant 0 : i32
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort dimension(1) outs(%[[ARG0]] : tensor<1x10xi32>)  {
// CHECK:         ^bb0(%[[ARG1:.+]]: i32, %{{.*}}: i32)
// CHECK:           %[[RES:.+]] = arith.cmpi slt, %[[ARG1]], %[[SCALAR]] : i32
// CHECK:           iree_linalg_ext.yield %[[RES]] : i1
// CHECK:         } -> tensor<1x10xi32>
// CHECK:         return %[[SORT]]

// -----

func.func @sort_argument_capture(%arg0: tensor<1x10xi32>, %arg1 : tensor<i32>) -> tensor<1x10xi32> {
  %1 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %2 = "mhlo.compare"(%arg2, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x10xi32>) -> tensor<1x10xi32>
  return %1 : tensor<1x10xi32>
}

// CHECK-LABEL: func.func @sort_argument_capture(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[SCALAR:.+]] = tensor.extract %[[ARG1]][] : tensor<i32>
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort dimension(1) outs(%[[ARG0]] : tensor<1x10xi32>)  {
// CHECK:         ^bb0(%[[ARG2:.+]]: i32, %{{.*}}: i32)
// CHECK:           %[[RES:.+]] = arith.cmpi slt, %[[ARG2]], %[[SCALAR]] : i32
// CHECK:           iree_linalg_ext.yield %[[RES]] : i1
// CHECK:         } -> tensor<1x10xi32>
// CHECK:         return %[[SORT]]

// -----

func.func @sort_2d(%arg0: tensor<16x32xi32>) -> (tensor<16x32xi32>) {
  %0 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<16x32xi32>) -> (tensor<16x32xi32>)
  return %0 : tensor<16x32xi32>
}
// CHECK-LABEL: func.func @sort_2d(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<16x32xi32>)
// CHECK:           ^bb0(%[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
// CHECK:             %[[CMP:.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG2]]
// CHECK:             iree_linalg_ext.yield %[[CMP]]
// CHECK:         return %[[SORT]]

// -----

func.func @sort_unsigned(%arg0: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %1 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = "mhlo.bitcast_convert"(%arg1) : (tensor<f32>) -> tensor<ui32>
    %3 = "mhlo.bitcast_convert"(%arg2) : (tensor<f32>) -> tensor<ui32>
    %4 = "mhlo.compare"(%2, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    "mhlo.return"(%4) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x5xf32>) -> tensor<1x5xf32>
  return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: func.func @sort_unsigned(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<1x5xf32>)
// CHECK:           ^bb0(%[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32)
// CHECK:             %[[CAST1:.+]] = arith.bitcast %[[ARG1]] : f32 to i32
// CHECK:             %[[CAST2:.+]] = arith.bitcast %[[ARG2]] : f32 to i32
// CHECK:             %[[CMP:.+]] = arith.cmpi ult, %[[CAST1]], %[[CAST2]] : i32
// CHECK:             iree_linalg_ext.yield %[[CMP]]
// CHECK:         return %[[SORT]]

// -----

func.func @sort_unsigned_cst_capture(%arg0: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %ui32 = mhlo.constant dense<2> : tensor<ui32>
  %1 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = "mhlo.bitcast_convert"(%arg1) : (tensor<f32>) -> tensor<ui32>
    %3 = "mhlo.compare"(%2, %ui32) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    "mhlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x5xf32>) -> tensor<1x5xf32>
  return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: func.func @sort_unsigned_cst_capture(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[UI32:.+]] = mhlo.constant dense<2> : tensor<ui32>
// CHECK:         %[[CONVERSION_CAST_CST:.+]] = tensor.bitcast %[[UI32]] : tensor<ui32> to tensor<i32>
// CHECK:         %[[EXTRACT_CST:.+]] = tensor.extract %[[CONVERSION_CAST_CST]][] : tensor<i32>
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<1x5xf32>)
// CHECK:           ^bb0(%[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32)
// CHECK:             %[[CAST1:.+]] = arith.bitcast %[[ARG1]] : f32 to i32
// CHECK:             %[[CMP:.+]] = arith.cmpi ult, %[[CAST1]], %[[EXTRACT_CST]] : i32
// CHECK:             iree_linalg_ext.yield %[[CMP]]
// CHECK:         return %[[SORT]]

// -----

// For testing that complex within an iree_linalg_ext.op gets lowered
func.func @sort_complex(%arg0: tensor<1x5xf32>, %arg1 : tensor<complex<f32>>) -> tensor<1x5xf32> {
  %ui32 = mhlo.constant dense<2> : tensor<ui32>
  %1 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = "mhlo.complex"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
    %3 = mhlo.add %2, %arg1 : tensor<complex<f32>>
    %4 = "mhlo.real"(%3) : (tensor<complex<f32>>) -> tensor<f32>
    %5 = "mhlo.imag"(%3) : (tensor<complex<f32>>) -> tensor<f32>
    %6 = "mhlo.compare"(%4, %5) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%6) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x5xf32>) -> tensor<1x5xf32>
  return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: func.func @sort_complex(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:  )
// CHECK:         %[[SORT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:    dimension(1)
// CHECK-SAME:    outs(%[[ARG0]] : tensor<1x5xf32>)
// CHECK:         ^bb0(%[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32)
// CHECK-NOT:       mhlo.complex
// CHECK:           %[[CMP:.+]] = arith.cmpf olt, %{{.+}}, %{{.+}} : f32
// CHECK:           iree_linalg_ext.yield %[[CMP]]
// CHECK:       return %[[SORT]]

// -----

func.func @topk(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> (tensor<128xi32>) {
  %0:2 = "mhlo.sort"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<128xi32>, tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  return %0#0 : tensor<128xi32>
}
// CHECK-LABEL: func.func @topk
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[SORT:.+]]:2 = iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs(%[[ARG0]], %[[ARG1]] : tensor<128xi32>, tensor<128xi32>)
// CHECK:           ^bb0(%[[ARG2:.+]]: i32, %[[ARG3:.+]]: i32, %{{.*}}: i32, %{{.*}}: i32)
// CHECK:             %[[CMP:.+]] = arith.cmpi sgt, %[[ARG2]], %[[ARG3]]
// CHECK:             iree_linalg_ext.yield %[[CMP]]
// CHECK:        return %[[SORT]]#0

// -----

func.func @scatter_update_scalar_1D(%arg0: tensor<8xi32>, %arg1: tensor<4x1xi32>,
    %arg2: tensor<4xi32>) -> tensor<8xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: func.func @scatter_update_scalar_1D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK:         %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:      unique_indices(true)
// CHECK-SAME:      ins(%[[ARG2]], %[[ARG1]] : tensor<4xi32>, tensor<4x1xi32>)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<8xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):
// CHECK:             iree_linalg_ext.yield %[[V1]]
// CHECK:         return %[[SCATTER]]

// -----

func.func @scatter_update_scalar_2D(%arg0: tensor<4x3xi32>, %arg1: tensor<3x2xi32>,
    %arg2: tensor<3xi32>) -> tensor<4x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {indices_are_sorted = false,
      scatter_dimension_numbers = #mhlo.scatter<
        inserted_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 1,
      >,
      unique_indices = true
  } : (tensor<4x3xi32>, tensor<3x2xi32>, tensor<3xi32>) -> tensor<4x3xi32>
  return %0 : tensor<4x3xi32>
}
// CHECK-LABEL: func.func @scatter_update_scalar_2D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK:         %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:      unique_indices(true)
// CHECK-SAME:      ins(%[[ARG2]], %[[ARG1]] : tensor<3xi32>, tensor<3x2xi32>)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<4x3xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):
// CHECK:             iree_linalg_ext.yield %[[V1]]
// CHECK:         return %[[SCATTER]]

// -----

func.func @scatter_update_slice_2D(%arg0: tensor<6x3xi32>, %arg1: tensor<2x1xi32>,
    %arg2: tensor<2x3xi32>) -> tensor<6x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  return %0 : tensor<6x3xi32>
}
// CHECK-LABEL: func.func @scatter_update_slice_2D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK:         %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:      unique_indices(true)
// CHECK-SAME:      ins(%[[ARG2]], %[[ARG1]] : tensor<2x3xi32>, tensor<2x1xi32>)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<6x3xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):
// CHECK:             iree_linalg_ext.yield %[[V1]]
// CHECK:         return %[[SCATTER]]

// -----

func.func @scatter_add_slice_2D(%arg0: tensor<6x3xi32>, %arg1: tensor<2x1xi32>,
    %arg2: tensor<2x3xi32>) -> tensor<6x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  return %0 : tensor<6x3xi32>
}
// CHECK-LABEL: func.func @scatter_add_slice_2D
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG2:[a-zA-Z0-9]+]]
// CHECK:         %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:      unique_indices(false)
// CHECK-SAME:      ins(%[[ARG2]], %[[ARG1]] : tensor<2x3xi32>, tensor<2x1xi32>)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<6x3xi32>)
// CHECK:           ^bb0(%[[V1:.+]]: i32, %[[V2:.+]]: i32):
//
//                   The order is reverse.
// CHECK:              %[[V3:.+]] = arith.addi %[[V2]], %[[V1]]
// CHECK:              iree_linalg_ext.yield %[[V3]]
// CHECK:         return %[[SCATTER]]

// -----

func.func @scatter_partial(%arg0: tensor<10x5xf32>, %arg1: tensor<3x1xi32>, %arg2: tensor<3x3xf32>) -> tensor<10x5xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<10x5xf32>, tensor<3x1xi32>, tensor<3x3xf32>) -> tensor<10x5xf32>
  return %0 : tensor<10x5xf32>
}

// CHECK-LABEL: func.func @scatter_partial
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK:         %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:      unique_indices(false)
// CHECK-SAME:      ins(%[[ARG2]], %[[ARG1]] : tensor<3x3xf32>, tensor<3x1xi32>)
// CHECK-SAME:      outs(%[[ARG0]] : tensor<10x5xf32>)
// CHECK:         return %[[SCATTER]]

// -----

func.func @rfft_1d(%input: tensor<8xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>
  } : (tensor<8xf32>) -> tensor<5xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<5xcomplex<f32>>) -> tensor<5xf32>
  %2 = "mhlo.imag"(%0) : (tensor<5xcomplex<f32>>) -> tensor<5xf32>
  return %1, %2 : tensor<5xf32>, tensor<5xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK:      func.func @rfft_1d
// CHECK-SAME:   %[[REAL:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[INDICES:.+]] = arith.constant dense<[0, 4, 2, 6, 1, 5, 3, 7]> : tensor<8xi32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
// CHECK:        %[[REORDERED:.+]] = linalg.generic
// CHECK-SAME:     {indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:     iterator_types = ["parallel"]
// CHECK-SAME:     ins(%[[INDICES]]
// CHECK-SAME:     outs(%[[EMPTY]]
// CHECK:        ^bb0(%[[IDX:.+]]: i32, %{{.+}}: f32):
// CHECK:          %[[IDXVAL:.+]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:          %[[LOAD:.+]] = tensor.extract %[[REAL]][%[[IDXVAL]]] : tensor<8xf32>
// CHECK:          linalg.yield %[[LOAD]] : f32
// CHECK-DAG:    %[[IMAG:.+]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[COEF_REAL:.+]] = arith.constant dense<{{.+}}> : tensor<1xf32>
// CHECK-DAG:    %[[COEF_IMAG:.+]] = arith.constant dense<{{.+}}> : tensor<1xf32>
// CHECK:        %[[R1:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:     ins(%[[C1]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:     outs(%[[REORDERED]], %[[IMAG]]
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[COEF_REAL:.+]] = arith.constant dense<{{.+}}> : tensor<2xf32>
// CHECK-DAG:    %[[COEF_IMAG:.+]] = arith.constant dense<{{.+}}> : tensor<2xf32>
// CHECK:        %[[R2:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:     ins(%[[C2]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:     outs(%[[R1]]#0, %[[R1]]#1
// CHECK-DAG:    %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:    %[[COEF_REAL:.+]] = arith.constant dense<{{.+}}> : tensor<4xf32>
// CHECK-DAG:    %[[COEF_IMAG:.+]] = arith.constant dense<{{.+}}> : tensor<4xf32>
// CHECK:        %[[R3:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:     ins(%[[C3]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:     outs(%[[R2]]#0, %[[R2]]#1
// CHECK:        %[[RES_REAL:.+]] = tensor.extract_slice %[[R3]]#0[0] [5] [1] : tensor<8xf32> to tensor<5xf32>
// CHECK:        %[[RES_IMAG:.+]] = tensor.extract_slice %[[R3]]#1[0] [5] [1] : tensor<8xf32> to tensor<5xf32>
// CHECK:        %{{.+}} = mhlo.complex %[[RES_REAL]], %[[RES_IMAG]]

// -----

func.func @rfft_2d(%input: tensor<4x8xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>) {
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>
  } : (tensor<4x8xf32>) -> tensor<4x5xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<4x5xcomplex<f32>>) -> tensor<4x5xf32>
  %2 = "mhlo.imag"(%0) : (tensor<4x5xcomplex<f32>>) -> tensor<4x5xf32>
  return %1, %2 : tensor<4x5xf32>, tensor<4x5xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:      func.func @rfft_2d
// CHECK-SAME:   %[[REAL:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[INDICES:.+]] = arith.constant dense<[0, 4, 2, 6, 1, 5, 3, 7]> : tensor<8xi32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
// CHECK:        %[[REORDERED:.+]] = linalg.generic
// CHECK-SAME:     {indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[INDICES]]
// CHECK-SAME:     outs(%[[EMPTY]]
// CHECK:        ^bb0(%[[IDX:.+]]: i32, %{{.+}}: f32):
// CHECK:          %[[I:.+]] = linalg.index 0
// CHECK:          %[[IDXVAL:.+]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:          %[[LOAD:.+]] = tensor.extract %[[REAL]][%[[I]], %[[IDXVAL]]] : tensor<4x8xf32>
// CHECK:          linalg.yield %[[LOAD]] : f32
// CHECK-DAG:    %[[IMAG:.+]] = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[COEF_REAL:.+]] = arith.constant dense<{{.+}}> : tensor<1xf32>
// CHECK-DAG:    %[[COEF_IMAG:.+]] = arith.constant dense<{{.+}}> : tensor<1xf32>
// CHECK:        %[[R1:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:     ins(%[[C1]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:     outs(%[[REORDERED]], %[[IMAG]]
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[COEF_REAL:.+]] = arith.constant dense<{{.+}}> : tensor<2xf32>
// CHECK-DAG:    %[[COEF_IMAG:.+]] = arith.constant dense<{{.+}}> : tensor<2xf32>
// CHECK:        %[[R2:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:     ins(%[[C2]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:     outs(%[[R1]]#0, %[[R1]]#1
// CHECK-DAG:    %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:    %[[COEF_REAL:.+]] = arith.constant dense<{{.+}}> : tensor<4xf32>
// CHECK-DAG:    %[[COEF_IMAG:.+]] = arith.constant dense<{{.+}}> : tensor<4xf32>
// CHECK:        %[[R3:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:     ins(%[[C3]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:     outs(%[[R2]]#0, %[[R2]]#1
// CHECK:        %[[RES_REAL:.+]] = tensor.extract_slice %[[R3]]#0[0, 0] [4, 5] [1, 1] : tensor<4x8xf32> to tensor<4x5xf32>
// CHECK:        %[[RES_IMAG:.+]] = tensor.extract_slice %[[R3]]#1[0, 0] [4, 5] [1, 1] : tensor<4x8xf32> to tensor<4x5xf32>
// CHECK:        %{{.+}} = mhlo.complex %[[RES_REAL]], %[[RES_IMAG]]

// -----

func.func @reverse_dim1(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %0 = "mhlo.reverse"(%arg0) {
    dimensions = dense<1> : tensor<1xi64>
  } : (tensor<3x5xi32>) -> tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
}
// CHECK-LABEL: func.func @reverse_dim1
// CHECK-SAME:   %[[IN:[a-zA-Z0-9]+]]
// CHECK:        %[[INIT:.+]] = tensor.empty() : tensor<3x5xi32>
// CHECK:        %[[REV:.+]] = iree_linalg_ext.reverse
// CHECK-SAME:     dimensions(dense<1> : tensor<1xi64>)
// CHECK-SAME:     ins(%[[IN]] : tensor<3x5xi32>)
// CHECK-SAME:     outs(%[[INIT]] : tensor<3x5xi32>) : tensor<3x5xi32>
// CHECK:        return %[[REV]]

// -----

func.func @reverse_unsigned(%arg0: tensor<3x5xui32>) -> tensor<3x5xui32> {
  %0 = "mhlo.reverse"(%arg0) {
    dimensions = dense<1> : tensor<1xi64>
  } : (tensor<3x5xui32>) -> tensor<3x5xui32>
  return %0 : tensor<3x5xui32>
}
// CHECK-LABEL: func.func @reverse_unsigned
// CHECK-SAME:   %[[IN:[a-zA-Z0-9]+]]
// CHECK:        %[[BITCAST:.+]] = tensor.bitcast %[[IN]] : tensor<3x5xui32> to tensor<3x5xi32>
// CHECK:        %[[INIT:.+]] = tensor.empty() : tensor<3x5xi32>
// CHECK:        %[[REV:.+]] = iree_linalg_ext.reverse
// CHECK-SAME:     dimensions(dense<1> : tensor<1xi64>)
// CHECK-SAME:     ins(%[[BITCAST]] : tensor<3x5xi32>)
// CHECK-SAME:     outs(%[[INIT]] : tensor<3x5xi32>) : tensor<3x5xi32>
// CHECK:        %[[BITCAST:.+]] = tensor.bitcast %[[REV]] : tensor<3x5xi32> to tensor<3x5xui32>
// CHECK:        return %[[BITCAST]]

// -----

func.func @reverse_multi_dim(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = "mhlo.reverse"(%arg0) {
    dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @reverse_multi_dim
// CHECK-SAME:   %[[IN:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.dim %[[IN]], %[[C0]]
// CHECK-DAG:    %[[D1:.+]] = tensor.dim %[[IN]], %[[C1]]
// CHECK:        %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xi32>
// CHECK:        %[[REV:.+]] = iree_linalg_ext.reverse
// CHECK-SAME:     dimensions(dense<[0, 1]> : tensor<2xi64>)
// CHECK-SAME:     ins(%[[IN]] : tensor<?x?xi32>)
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xi32>) : tensor<?x?xi32>
// CHECK:        return %[[REV]]

// -----

func.func @chlo_top_k_int(%arg : tensor<16x16xi32>) -> (tensor<16x8xi32>, tensor<16x8xi32>) {
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xi32> -> (tensor<16x8xi32>, tensor<16x8xi32>)
  return %1#0, %1#1 : tensor<16x8xi32>, tensor<16x8xi32>
}

// CHECK:       func.func @chlo_top_k_int
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:        %[[D2:.+]] = tensor.empty() : tensor<16x8xi32>
// CHECK:        %[[D3:.+]] = tensor.empty() : tensor<16x8xi32>
// CHECK-DAG:    %[[CNEG:.+]] = arith.constant -2147483648 : i32
// CHECK-DAG:    %[[CPOS:.+]] = arith.constant 2147483647 : i32
// CHECK-DAG:    %[[D4:.+]] = linalg.fill ins(%[[CNEG]] : i32) outs(%[[D2]]
// CHECK-DAG:    %[[D5:.+]] = linalg.fill ins(%[[CPOS]] : i32) outs(%[[D3]]
// CHECK:        %[[D6:.+]]:2 = iree_linalg_ext.topk
// CHECK-SAME:     dimension(1)
// CHECK-SAME:     ins(%[[ARG0]]
// CHECK-SAME:     outs(%[[D4]], %[[D5]]
// CHECK:        ^bb0(%[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
// CHECK:        %[[D7:.+]] = arith.cmpi sge, %[[ARG1]], %[[ARG2]] : i32
// CHECK:        iree_linalg_ext.yield %[[D7]] : i1
// CHECK:        return %[[D6]]#0, %[[D6]]#1

// -----

func.func @chlo_top_k_float(%arg : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  return %1#0, %1#1 : tensor<16x8xf32>, tensor<16x8xi32>
}

// CHECK:       func.func @chlo_top_k_float
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:        %[[D2:.+]] = tensor.empty() : tensor<16x8xf32>
// CHECK:        %[[D3:.+]] = tensor.empty() : tensor<16x8xi32>
// CHECK-DAG:    %[[CNEG:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:    %[[CPOS:.+]] = arith.constant 2147483647 : i32
// CHECK-DAG:    %[[D4:.+]] = linalg.fill ins(%[[CNEG]] : f32) outs(%[[D2]]
// CHECK-DAG:    %[[D5:.+]] = linalg.fill ins(%[[CPOS]] : i32) outs(%[[D3]]
// CHECK:        %[[D6:.+]]:2 = iree_linalg_ext.topk
// CHECK-SAME:     dimension(1)
// CHECK-SAME:     ins(%[[ARG0]]
// CHECK-SAME:     outs(%[[D4]], %[[D5]]
// CHECK:        ^bb0(%[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32)
// CHECK:        %[[D7:.+]] = arith.cmpf ogt, %[[ARG1]], %[[ARG2]] : f32
// CHECK:        iree_linalg_ext.yield %[[D7]] : i1
// CHECK:        return %[[D6]]#0, %[[D6]]#1
