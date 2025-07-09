// RUN: iree-opt --split-input-file %s | FileCheck %s

util.func public @workgroup_count_splitk(%arg0: index, %arg1: index, %arg2: index) -> tensor<?x?x?x?x?xf32> {
  %0 = flow.dispatch.workgroups[%arg0, %arg1, %arg2](%arg0, %arg1, %arg2) : (index, index, index) -> tensor<?x?x?x?x?xf32>{%arg0, %arg1, %arg2, %arg2, %arg0} =
      (%arg3: index, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>) {
    flow.return
  } count(%arg3: index, %arg4: index, %arg5: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg3, %arg4, %arg5
    %result_x, %result_y, %result_z = iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier(%x, %y, %z), %arg3, %arg4, %arg5
    flow.return %result_x, %result_y, %result_z : index, index, index
  }
  util.return %0 : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: util.func public @workgroup_count_splitk(
//       CHECK:   count(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice
//       CHECK:   %[[X0:.+]], %[[Y0:.+]], %[[Z0:.+]] = iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier
//  CHECK-SAME:   (%[[X]], %[[Y]], %[[Z]]), %[[ARG0]], %[[ARG1]], %[[ARG2]]
