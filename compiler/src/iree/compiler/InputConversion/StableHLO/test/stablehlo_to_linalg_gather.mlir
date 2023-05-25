// RUN: iree-opt %s --iree-stablehlo-to-linalg --split-input-file \
// RUN:   --canonicalize | FileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK:      func.func @gather(
// CHECK-SAME:     %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:     %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME: )
func.func @gather(%operand : tensor<1x4x8xi32>, %start_indices : tensor<1x8x2xi32>) -> tensor<1x8x8xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>,
    someattr
  } : (tensor<1x4x8xi32>, tensor<1x8x2xi32>) -> tensor<1x8x8xi32>
  func.return %res : tensor<1x8x8xi32>
}

// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<1x8x8xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           indexing_maps = [#[[MAP0]]],
// CHECK-SAME:           iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:           outs(%[[INIT]] : tensor<1x8x8xi32>)
// CHECK-SAME:           {someattr}
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[IDX1]], %[[C0]]] : tensor<1x8x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[IDX1]], %[[C1]]] : tensor<1x8x2xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]]  : index
// CHECK-DAG:         %[[IN0:.+]] = arith.minsi %[[CLAMP0]], %[[C0]]
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.minsi %[[CLAMP1]], %[[C3]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IDX2]]] : tensor<1x4x8xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK-DAG:       return %[[RES]]

// -----

// CHECK-LABEL:   func.func @gather_unsigned_index(
func.func @gather_unsigned_index(
    %operand : tensor<1x4x8xi32>, %start_indices : tensor<1x8x2xui32>)
    -> tensor<1x8x8xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>,
    someattr
  } : (tensor<1x4x8xi32>, tensor<1x8x2xui32>) -> tensor<1x8x8xi32>
  func.return %res : tensor<1x8x8xi32>
}

// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK:           %[[S0_INT:.+]] = tensor.extract {{.*}}[{{.*}}, %[[C0]]]
// CHECK:           arith.index_castui %[[S0_INT]] : i32 to index
// CHECK:           %[[S1_INT:.+]] = tensor.extract {{.*}}[{{.*}}, %[[C1]]]
// CHECK:           arith.index_castui %[[S1_INT]] : i32 to index

// -----

// CHECK-LABEL:   func @gather_unsigned(
func.func @gather_unsigned(%operand : tensor<1x4x8xui32>, %start_indices : tensor<1x8x2xi32>) -> tensor<1x8x8xui32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<1x4x8xui32>, tensor<1x8x2xi32>) -> tensor<1x8x8xui32>
  func.return %res : tensor<1x8x8xui32>
}

// CHECK:           linalg.generic
// CHECK-SAME:           outs(%{{.*}} : tensor<1x8x8xi32>)

// -----

// CHECK-LABEL:   func.func @gather_no_collapse(
// CHECK-SAME:         %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:         %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
func.func @gather_no_collapse(%operand : tensor<6x3xi32>, %start_indices : tensor<5x2xi32>) -> tensor<5x4x2xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[4, 2]> : tensor<2xi64>
  } : (tensor<6x3xi32>, tensor<5x2xi32>) -> tensor<5x4x2xi32>
  func.return %res : tensor<5x4x2xi32>
}

// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<5x4x2xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<5x4x2xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C0]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C1]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[C2]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX1]] : index
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP1_1:.+]] = arith.minsi %[[CLAMP1]], %[[C1]]
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[CLAMP1_1]], %[[IDX2]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]]] : tensor<6x3xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]


// -----

func.func @gather_max_offset(%operand : tensor<?x?x?xi32>, %start_indices : tensor<5x2xi32>) -> tensor<2x3x4x5xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [0, 1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[2, 3, 4]> : tensor<3xi64>
  } : (tensor<?x?x?xi32>, tensor<5x2xi32>) -> tensor<2x3x4x5xi32>
  func.return %res : tensor<2x3x4x5xi32>
}

// CHECK-LABEL:   func @gather_max_offset(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<2x3x4x5xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<2x3x4x5xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[IDX3:.+]] = linalg.index 3
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX3]], %[[C0]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX3]], %[[C1]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C2]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
// CHECK-DAG:         %[[L1:.+]] = arith.subi %[[D1]], %[[C3]]
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP1_1:.+]] = arith.minsi %[[CLAMP1]], %[[L1]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[CLAMP1_1]], %[[IDX1]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IDX2]]] : tensor<?x?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

func.func @gather_reorder_start_index(%operand : tensor<6x3x2x7xi32>, %start_indices : tensor<5x4xi32>) -> tensor<5x2x4xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 2],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [3, 1, 2, 0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 2, 1, 4]> : tensor<4xi64>
  } : (tensor<6x3x2x7xi32>, tensor<5x4xi32>) -> tensor<5x2x4xi32>
  func.return %res : tensor<5x2x4xi32>
}

// CHECK-LABEL:   func @gather_reorder_start_index(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[C5:.+]] = arith.constant 5
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<5x2x4xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<5x2x4xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C0]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C1]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[S2_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C2]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S2:.+]] = arith.index_cast %[[S2_INT]] : i32 to index
// CHECK-DAG:         %[[S3_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C3]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S3:.+]] = arith.index_cast %[[S3_INT]] : i32 to index
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP1_1:.+]] = arith.minsi %[[CLAMP1]], %[[C1]]
// CHECK-DAG:         %[[CLAMP2:.+]] = arith.maxsi %[[S2]], %[[C0]]  : index
// CHECK-DAG:         %[[IN2:.+]] = arith.minsi %[[CLAMP2]], %[[C1]]
// CHECK-DAG:         %[[CLAMP3:.+]] = arith.maxsi %[[S3]], %[[C0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.minsi %[[CLAMP3]], %[[C5]]
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[CLAMP1_1]], %[[IDX1]] : index
// CHECK-DAG:         %[[IN3:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX2]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IN2]], %[[IN3]]] : tensor<6x3x2x7xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

// CHECK-LABEL:   func.func @gather_implicit_trailing_dim(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
func.func @gather_implicit_trailing_dim(%operand : tensor<?x?xi32>, %start_indices : tensor<5x2xi32>) -> tensor<3x4x5x2xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 2,
      offset_dims = [0, 1],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 4]> : tensor<2xi64>
  } : (tensor<?x?xi32>, tensor<5x2xi32>) -> tensor<3x4x5x2xi32>
  func.return %res : tensor<3x4x5x2xi32>
}

// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<3x4x5x2xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<3x4x5x2xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[IDX3:.+]] = linalg.index 3
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[IDX3]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IDX1]]] : tensor<?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

// CHECK-LABEL:   func.func @gather_non_static(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
func.func @gather_non_static(%operand : tensor<?x?xi32>, %start_indices : tensor<?x?xi32>) -> tensor<3x4x?xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [0, 1],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 4]> : tensor<2xi64>
  } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<3x4x?xi32>
  func.return %res : tensor<3x4x?xi32>
}

// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[DYN_DIM:.+]] = tensor.dim %[[START_INDICES]], %[[C0]]
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty(%[[DYN_DIM]]) : tensor<3x4x?xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<3x4x?xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[C0]]] : tensor<?x?xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IDX1]]] : tensor<?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

// CHECK-LABEL:   func.func @gather_unranked(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
func.func @gather_unranked(%operand : tensor<*xi32>, %start_indices : tensor<?x?xi32>) -> tensor<?x?x?xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [0, 1],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 4]> : tensor<2xi64>
  } : (tensor<*xi32>, tensor<?x?xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[RES_DIM2:.+]] = tensor.dim %[[START_INDICES]], %[[C0]]
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty(%[[RES_DIM2]])
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[C0]]] : tensor<?x?xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK-DAG:         %[[OPERAND_CASTED:.+]] = tensor.cast %[[OPERAND]] : tensor<*xi32> to tensor<?x?xi32>
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND_CASTED]][%[[IN0]], %[[IDX1]]] : tensor<?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           %[[CAST:.+]] = tensor.cast %[[RES]]
// CHECK:           return %[[CAST]]
