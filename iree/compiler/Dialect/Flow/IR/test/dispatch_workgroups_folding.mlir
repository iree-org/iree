// RUN: iree-opt -allow-unregistered-dialect -split-input-file -canonicalize %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @workgroupStaticShapeDims
func @workgroupStaticShapeDims(%arg0 : tensor<?x4xf32>) -> tensor<4x?xf32> {
  %c128 = constant 128 : index
  %x = constant 100 : index
  %y = constant 50 : index
  // CHECK: flow.dispatch.workgroups
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<?x4xf32>{%c128}) -> (tensor<4x?xf32>{%c128}) = (
    // CHECK-NEXT: (%[[ARG0:.+]]: !flow.dispatch.tensor<readonly:?x4xf32>,
    %arg0_capture: !flow.dispatch.tensor<readonly:?x4xf32>,
    // CHECK-SAME:  %[[RET0:.+]]: !flow.dispatch.tensor<writeonly:4x?xf32>)
    %ret0: !flow.dispatch.tensor<writeonly:4x?xf32>
  ) {
    // CHECK: %[[DIM_4:.+]] = constant 4 : index

    // CHECK: %[[ARG0_SHAPE:.+]] = flow.dispatch.shape %[[ARG0]]
    %arg0_shape = flow.dispatch.shape %arg0_capture : !flow.dispatch.tensor<readonly:?x4xf32> -> !shapex.ranked_shape<[?,4]>
    // CHECK: %[[ARG0_DIM0:.+]] = shapex.ranked_dim %[[ARG0_SHAPE]][0]
    %arg0_dim0 = shapex.ranked_dim %arg0_shape[0] : !shapex.ranked_shape<[?,4]> -> index
    %arg0_dim1 = shapex.ranked_dim %arg0_shape[1] : !shapex.ranked_shape<[?,4]> -> index
    // CHECK: "test.sink"(%[[ARG0_DIM0]], %[[DIM_4]])
    "test.sink"(%arg0_dim0, %arg0_dim1) : (index, index) -> ()

    // CHECK: %[[RET0_SHAPE:.+]] = flow.dispatch.shape %[[RET0]]
    %ret0_shape = flow.dispatch.shape %ret0 : !flow.dispatch.tensor<writeonly:4x?xf32> -> !shapex.ranked_shape<[4,?]>
    %ret0_dim0 = shapex.ranked_dim %ret0_shape[0] : !shapex.ranked_shape<[4,?]> -> index
    // CHECK: %[[RET0_DIM1:.+]] = shapex.ranked_dim %[[RET0_SHAPE]][1]
    %ret0_dim1 = shapex.ranked_dim %ret0_shape[1] : !shapex.ranked_shape<[4,?]> -> index
    // CHECK: "test.sink"(%[[DIM_4]], %[[RET0_DIM1]])
    "test.sink"(%ret0_dim0, %ret0_dim1) : (index, index) -> ()

    flow.return
  }
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @workgroupRankFolding
func @workgroupRankFolding(%arg0 : tensor<?x4xf32>) -> tensor<4x?xf32> {
  %c128 = constant 128 : index
  %x = constant 100 : index
  %y = constant 50 : index
  // CHECK: flow.dispatch.workgroups
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<?x4xf32>{%c128}) -> (tensor<4x?xf32>{%c128}) = (
    %arg0_capture: !flow.dispatch.tensor<readonly:?x4xf32>,
    %ret0: !flow.dispatch.tensor<writeonly:4x?xf32>
  ) {
    // CHECK: %[[RANK:.+]] = constant 2 : index
    %workgroup_rank = flow.dispatch.workgroup.rank : index
    // CHECK-NEXT: "test.sink"(%[[RANK]])
    "test.sink"(%workgroup_rank) : (index) -> ()
    flow.return
  }
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @convertDimOfDispatchInputLoadToDispatchShape
// CHECK-SAME:    %[[ARG:.*]]: !flow.dispatch.tensor<readonly:?xf32>) {
func @convertDimOfDispatchInputLoadToDispatchShape(%arg0: !flow.dispatch.tensor<readonly:?xf32>) {
  // CHECK-NEXT: %[[RANKED_SHAPE:.*]] = flow.dispatch.shape %[[ARG]]
  // CHECK-NEXT: %[[DIM:.*]] = shapex.ranked_dim %[[RANKED_SHAPE]][0]
  // CHECK-NEXT: "test.sink"(%[[DIM]]) : (index) -> ()
  %tensor = flow.dispatch.tensor.load %arg0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:?xf32> -> tensor<?xf32>
  %c0 = constant 0 : index
  %dim = memref.dim %tensor, %c0 : tensor<?xf32>
  "test.sink"(%dim) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: @inlineWithTiedResults1
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
func @inlineWithTiedResults1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK-NOT: constant 128
  %cst = constant 128 : index
  // CHECK-DAG: %[[X:.+]] = constant 100
  %x = constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = constant 50
  %y = constant 50 : index
  //      CHECK: flow.dispatch.workgroups[%[[X]], %[[Y]]](%[[ARG0]]) : (tensor<1x4xf32>) -> %[[ARG0]] =
  // CHECK-NEXT:   (%[[ARG0_INNER:.+]]: !flow.dispatch.tensor<readwrite:1x4xf32>)
  %0 = flow.dispatch.workgroups[%x, %y](%cst, %arg0) : (index, tensor<1x4xf32>) -> %arg0 = (
    %cst_capture: index,
    %arg0_capture: !flow.dispatch.tensor<readwrite:1x4xf32>
  ) {
    //      CHECK: %[[INLINED_CST:.+]] = constant 128 : index
    // CHECK-NEXT: "test.sink"(%[[INLINED_CST]])
    "test.sink"(%cst_capture) : (index) -> ()
    // CHECK-NEXT: "test.sink"(%[[ARG0_INNER]])
    "test.sink"(%arg0_capture) : (!flow.dispatch.tensor<readwrite:1x4xf32>) -> ()
    flow.return
  }
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @inlineWithTiedResults2
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
func @inlineWithTiedResults2(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK-NOT: constant 128
  %cst = constant 128 : index
  // CHECK-DAG: %[[X:.+]] = constant 100
  %x = constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = constant 50
  %y = constant 50 : index
  //      CHECK: flow.dispatch.workgroups[%[[X]], %[[Y]]](%[[ARG0]]) : (tensor<1x4xf32>) -> %[[ARG0]] =
  // CHECK-NEXT:   (%[[ARG0_INNER:.+]]: !flow.dispatch.tensor<readwrite:1x4xf32>)
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %cst) : (tensor<1x4xf32>, index) -> %arg0 = (
    %arg0_capture: !flow.dispatch.tensor<readwrite:1x4xf32>,
    %cst_capture: index
  ) {
    //      CHECK: %[[INLINED_CST:.+]] = constant 128 : index
    // CHECK-NEXT: "test.sink"(%[[INLINED_CST]])
    "test.sink"(%cst_capture) : (index) -> ()
    // CHECK-NEXT: "test.sink"(%[[ARG0_INNER]])
    "test.sink"(%arg0_capture) : (!flow.dispatch.tensor<readwrite:1x4xf32>) -> ()
    flow.return
  }
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dontInlineReadWrite
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
func @dontInlineReadWrite(%arg0: tensor<1x4xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[CST:.+]] = constant dense<0.000000e+00> : tensor<4x8xf32>
  %cst = constant dense<0.0> : tensor<4x8xf32>
  %x = constant 100 : index
  %y = constant 50 : index
  //      CHECK: flow.dispatch.workgroups[{{.+}}](%[[ARG0]], %[[CST]]) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<readonly:1x4xf32>, %{{.+}}: !flow.dispatch.tensor<readwrite:4x8xf32>)
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %cst) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst = (
    %arg0_capture: !flow.dispatch.tensor<readonly:1x4xf32>,
    %arg1_capture: !flow.dispatch.tensor<readwrite:4x8xf32>
  ) {
    "test.sink"(%arg0_capture) : (!flow.dispatch.tensor<readonly:1x4xf32>) -> ()
    %load = flow.dispatch.tensor.load %arg1_capture, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readwrite:4x8xf32> -> tensor<4x8xf32>
    %0 = "test.do_work"(%load) : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %0, %arg1_capture, offsets=[], sizes=[], strides=[] : tensor<4x8xf32> -> !flow.dispatch.tensor<readwrite:4x8xf32>
    flow.return
  }
  return %0 : tensor<4x8xf32>
}
