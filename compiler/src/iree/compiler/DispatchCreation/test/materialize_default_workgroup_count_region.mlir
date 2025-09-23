// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-materialize-default-workgroup-count-region))" --split-input-file --verify-diagnostics %s | FileCheck %s

// Basic test of creating a default workgroup count region when none exists.
util.func @test_simple_creation(%arg0 : index, %arg1 : index) {
  %0 = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1)
      : (index, index) -> (tensor<?x?xf32>{%arg0, %arg1}) =
    (%arg0_capture: index, %arg1_capture : index,
     %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    %1 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
    iree_tensor_ext.dispatch.tensor.store %1, %output,
        offsets = [0, 0], sizes = [%arg0_capture, %arg1_capture], strides = [1, 1]
        : tensor<?x?xf32> ->
          !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg0_capture, %arg1_capture}
    flow.return
  }
  util.return
}
// CHECK-LABEL: @test_simple_creation
//       CHECK:   count(%[[WL0:[a-zA-Z0-9_]+]]: index, %[[WL1:[a-zA-Z0-9_]+]]: index)
//       CHECK:     %[[X:[a-zA-Z0-9_]+]], %[[Y:[a-zA-Z0-9_]+]], %[[Z:[a-zA-Z0-9_]+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[WL0]], %[[WL1]])
//       CHECK:     flow.return %[[X]], %[[Y]], %[[Z]]

// -----

// Test that when a count region exists nothing is done.
util.func @test_existing_count_region(%arg0 : index, %arg1 : index) {
  %0 = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1)
      : (index, index) -> (tensor<?x?xf32>{%arg0, %arg1}) =
    (%arg0_capture: index, %arg1_capture : index,
     %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    %1 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
    iree_tensor_ext.dispatch.tensor.store %1, %output,
        offsets = [0, 0], sizes = [%arg0_capture, %arg1_capture], strides = [1, 1]
        : tensor<?x?xf32> ->
          !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg0_capture, %arg1_capture}
    flow.return
  } count(%workload0 : index, %workload1 : index) -> (index, index, index) {
    %c1 = arith.constant 1: index
    %1 = arith.muli %workload0, %workload1 : index
    flow.return %1, %c1, %c1 : index, index, index
  }
  util.return
}
// CHECK-LABEL: @test_existing_count_region
//       CHECK:   count(%[[WL0:[a-zA-Z0-9_]+]]: index, %[[WL1:[a-zA-Z0-9_]+]]: index)
//       CHECK:     %[[C1:.+]] = arith.constant 1 : index
//       CHECK:     %[[MUL:.+]] = arith.muli %[[WL0]], %[[WL1]]
//       CHECK:     flow.return %[[MUL]], %[[C1]], %[[C1]]

// -----

// Check the addition of the split-reduction modified.
util.func @test_split_reduction_modified(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1, %arg2)
      : (index, index, index) -> (tensor<?x?xf32>{%arg0, %arg1}) =
    (%arg0_capture: index, %arg1_capture : index, %arg2_capture : index,
     %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    %c0 = arith.constant 0 : index
    %1 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
    %2 = scf.forall (%iv0) in (%arg2_capture) shared_outs(%init = %1) -> tensor<?x?xf32> {
      %3 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %init[0, 0] [%arg0_capture, %arg1_capture] [1, 1]
            : tensor<?x?xf32> into tensor<?x?xf32>
      }
    } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
    iree_tensor_ext.dispatch.tensor.store %2, %output,
        offsets = [0, 0], sizes = [%arg0_capture, %arg1_capture], strides = [1, 1]
        : tensor<?x?xf32> ->
          !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg0_capture, %arg1_capture}
    flow.return
  }
  util.return
}
// CHECK-LABEL: @test_split_reduction_modified
//       CHECK:   count(%[[WL0:[a-zA-Z0-9_]+]]: index, %[[WL1:[a-zA-Z0-9_]+]]: index, %[[WL2:[a-zA-Z0-9_]+]]: index)
//       CHECK:     %[[X0:[a-zA-Z0-9_]+]], %[[Y0:[a-zA-Z0-9_]+]], %[[Z0:[a-zA-Z0-9_]+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[WL0]], %[[WL1]], %[[WL2]])
//       CHECK:     %[[X1:[a-zA-Z0-9_]+]], %[[Y1:[a-zA-Z0-9_]+]], %[[Z1:[a-zA-Z0-9_]+]] = iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier workgroups(%[[X0]], %[[Y0]], %[[Z0]]) workload(%[[WL0]], %[[WL1]], %[[WL2]])
//       CHECK:     flow.return %[[X1]], %[[Y1]], %[[Z1]]

// -----

// Check that without the split-reduction-mapping having scf.forall raises an error.
util.func @error_no_mapping(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error @below {{unhandled scf.forall op that doesnt have a mapping of `[#iree_linalg_ext.split_reduction_mapping]`}}
  %0 = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1, %arg2)
      : (index, index, index) -> (tensor<?x?xf32>{%arg0, %arg1}) =
    (%arg0_capture: index, %arg1_capture : index, %arg2_capture : index,
     %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    %c0 = arith.constant 0 : index
    %1 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
    %2 = scf.forall (%iv0) in (%arg2_capture) shared_outs(%init = %1) -> tensor<?x?xf32> {
      %3 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %init[0, 0] [%arg0_capture, %arg1_capture] [1, 1]
            : tensor<?x?xf32> into tensor<?x?xf32>
      }
    }
    iree_tensor_ext.dispatch.tensor.store %2, %output,
        offsets = [0, 0], sizes = [%arg0_capture, %arg1_capture], strides = [1, 1]
        : tensor<?x?xf32> ->
          !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg0_capture, %arg1_capture}
    flow.return
  }
  util.return
}

// -----

// Check that mutliple scf.forall loops within dispatch is not supported.
util.func @error_multiple_scf_forall(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error @below {{unhandled multiple scf.forall ops in a dispatch}}
  %0:2 = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1, %arg2)
      : (index, index, index) -> (tensor<?x?xf32>{%arg0, %arg1}, tensor<?x?xf32>{%arg0, %arg1}) =
    (%arg0_capture: index, %arg1_capture : index, %arg2_capture : index,
     %output0 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>,
     %output1 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    %c0 = arith.constant 0 : index
    %1 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
    %2 = scf.forall (%iv0) in (%arg2_capture) shared_outs(%init = %1) -> tensor<?x?xf32> {
      %3 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %init[0, 0] [%arg0_capture, %arg1_capture] [1, 1]
            : tensor<?x?xf32> into tensor<?x?xf32>
      }
    }
    %4 = scf.forall (%iv0) in (%arg2_capture) shared_outs(%init = %1) -> tensor<?x?xf32> {
      %5 = tensor.empty(%arg0_capture, %arg1_capture) : tensor<?x?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %init[0, 0] [%arg0_capture, %arg1_capture] [1, 1]
            : tensor<?x?xf32> into tensor<?x?xf32>
      }
    }
    iree_tensor_ext.dispatch.tensor.store %2, %output0,
        offsets = [0, 0], sizes = [%arg0_capture, %arg1_capture], strides = [1, 1]
        : tensor<?x?xf32> ->
          !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg0_capture, %arg1_capture}
    iree_tensor_ext.dispatch.tensor.store %4, %output1,
        offsets = [0, 0], sizes = [%arg0_capture, %arg1_capture], strides = [1, 1]
        : tensor<?x?xf32> ->
          !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg0_capture, %arg1_capture}
    flow.return
  }
  util.return
}
