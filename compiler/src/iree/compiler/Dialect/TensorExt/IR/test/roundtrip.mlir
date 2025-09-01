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
