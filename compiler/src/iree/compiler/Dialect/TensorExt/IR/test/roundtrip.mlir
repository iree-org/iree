// RUN: iree-opt --split-input-file %s | FileCheck %s

util.func public @workgroup_count_splitk(%arg0: index, %arg1: index, %arg2: index) -> tensor<?x?x?x?x?xf32> {
  %0 = flow.dispatch.workgroups[%arg0, %arg1, %arg2](%arg0, %arg1, %arg2) : (index, index, index) -> tensor<?x?x?x?x?xf32>{%arg0, %arg1, %arg2, %arg2, %arg0} =
      (%arg3: index, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>) {
    flow.return
  } count(%arg3: index, %arg4: index, %arg5: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg3, %arg4, %arg5)
    %result_x, %result_y, %result_z = iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier workgroups(%x, %y, %z) workload(%arg3, %arg4, %arg5)
    flow.return %result_x, %result_y, %result_z : index, index, index
  }
  util.return %0 : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: util.func public @workgroup_count_splitk(
//       CHECK:   count(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG1]], %[[ARG2]])
//       CHECK:   %[[X0:.+]], %[[Y0:.+]], %[[Z0:.+]] = iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier
//  CHECK-SAME:   workgroups(%[[X]], %[[Y]], %[[Z]])
//  CHECK-SAME:   workload(%[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

// CHECK-LABEL: @workgroup_count_no_args
util.func public @workgroup_count_no_args() {
  %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
  %result_x, %result_y, %result_z = iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier workgroups(%x, %y, %z) workload()
  %a, %b, %c = iree_tensor_ext.dispatch.workgroup_count_from_dag_root()
  util.return
}

// -----

// CHECK-LABEL: @tensorBitCast
util.func public @tensorBitCast(%arg0: tensor<16xi32>) -> tensor<4x8xi16> {
  // CHECK-NEXT: %0 = iree_tensor_ext.bitcast %arg0 : tensor<16xi32> -> tensor<4x8xi16>
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<16xi32> -> tensor<4x8xi16>
  util.return %0 : tensor<4x8xi16>
}

// -----

// CHECK-LABEL: @tensorBitCastDynamic
util.func public @tensorBitCastDynamic(%arg0: tensor<?x16xi32>, %arg1: index, %arg2: index, %arg3:index) -> tensor<?x?x4x8xi16> {
  // CHECK-NEXT: %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x16xi32>{%arg1} -> tensor<?x?x4x8xi16>{%arg2, %arg3}
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x16xi32>{%arg1} -> tensor<?x?x4x8xi16>{%arg2, %arg3}
  util.return %0 : tensor<?x?x4x8xi16>
}

// -----

// CHECK-LABEL: @barrier_start_static
util.func public @barrier_start_static(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: iree_tensor_ext.compute_barrier.start
  %0 = iree_tensor_ext.compute_barrier.start %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  util.return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @barrier_start_dynamic
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index
util.func public @barrier_start_dynamic(%arg0: tensor<?x?xf32>, %dim0: index, %dim1: index) -> tensor<?x?xf32> {
  // CHECK: iree_tensor_ext.compute_barrier.start %arg0 : tensor<?x?xf32>{%[[DIM0]], %[[DIM1]]} -> tensor<?x?xf32>
  %0 = iree_tensor_ext.compute_barrier.start %arg0 : tensor<?x?xf32>{%dim0, %dim1} -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @barrier_end_static
util.func public @barrier_end_static(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: iree_tensor_ext.compute_barrier.end
  %0 = iree_tensor_ext.compute_barrier.end %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  util.return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @barrier_end_dynamic
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index
util.func public @barrier_end_dynamic(%arg0: tensor<?x?xf32>, %dim0: index, %dim1: index) -> tensor<?x?xf32> {
  // CHECK: iree_tensor_ext.compute_barrier.end %[[ARG0]] : tensor<?x?xf32>{%[[DIM0]], %[[DIM1]]} -> tensor<?x?xf32>
  %0 = iree_tensor_ext.compute_barrier.end %arg0 : tensor<?x?xf32>{%dim0, %dim1} -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// -----

// Check RaggedTensorAttr.

util.func public @raggedTensorAttrTest(%arg0 : tensor<?x?xf32, #iree_tensor_ext.ragged_shape<0>>) {
  util.return
}
// CHECK-LABEL: @raggedTensorAttrTest
//  CHECK-SAME:     #iree_tensor_ext.ragged_shape<0>

// -----

// Check static iree_tensor_ext.cast_to_ragged.

util.func public @castToRaggedStatic(%source : tensor<10x20x30xf32>,
    %columnLengths : tensor<4xindex>) -> tensor<10x3x?x30xf32, #iree_tensor_ext.ragged_shape<1>> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<10x20x30xf32>, tensor<4xindex>) -> tensor<10x3x?x30xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<10x3x?x30xf32, #iree_tensor_ext.ragged_shape<1>>
}
// CHECK-LABEL: @castToRaggedStatic
//       CHECK:     iree_tensor_ext.cast_to_ragged_shape

// -----

// Check dynamic iree_tensor_ext.cast_to_ragged.

util.func public @castToRaggedDynamic(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>, %numRaggedRows : index)
    -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %source, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %source, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %source, %c2 : tensor<?x?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>)
      -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}
// CHECK-LABEL: @castToRaggedDynamic
//       CHECK:     iree_tensor_ext.cast_to_ragged_shape

// -----

// Check memref-type iree_tensor_ext.cast_to_ragged.

util.func public @castToRaggedDynamicMemRef(%source : memref<?x?x?xf32>,
    %columnLengths : memref<?xindex>, %numRaggedRows : index)
    -> memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = memref.dim %source, %c0 : memref<?x?x?xf32>
  %d1 = memref.dim %source, %c1 : memref<?x?x?xf32>
  %d2 = memref.dim %source, %c2 : memref<?x?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (memref<?x?x?xf32>{%d0, %d1, %d2}, memref<?xindex>)
      -> memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}
// CHECK-LABEL: @castToRaggedDynamic
//       CHECK:     iree_tensor_ext.cast_to_ragged_shape

// -----

// Check for static number of ragged rows.

util.func public @staticNumRaggedRows(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %maxColumnLength : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}
// CHECK-LABEL: @staticNumRaggedRows
//       CHECK:     iree_tensor_ext.cast_to_ragged_shape

// -----

util.func public @staticAvgRaggedColumnLengths(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}
// CHECK-LABEL: @staticAvgRaggedColumnLengths
//       CHECK:     iree_tensor_ext.cast_to_ragged_shape

// -----

// Check for dynamically specified `avg_ragged_column_length`.

util.func public @dynamicAvgRaggedColumnLengths(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %avgColumnLength : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      avg_ragged_column_length(%avgColumnLength)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}
// CHECK-LABEL: @dynamicAvgRaggedColumnLengths
//       CHECK:     iree_tensor_ext.cast_to_ragged_shape
