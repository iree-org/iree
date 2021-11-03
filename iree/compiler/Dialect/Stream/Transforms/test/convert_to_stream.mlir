// RUN: iree-opt -split-input-file -iree-stream-conversion %s | IreeFileCheck %s

// CHECK: stream.executable private @executable
flow.executable private @executable {
  // CHECK: stream.executable.export public @dispatch
  flow.dispatch.entry public @dispatch attributes {workgroup_rank = 3 : index}
  builtin.module {
    // CHECK: func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index, %arg3: index)
    func @dispatch(%arg0: !flow.dispatch.tensor<readonly:?x4xf32>, %arg1: !flow.dispatch.tensor<writeonly:?xf32>,
                   %arg0_dim0: index, %arg1_dim0: index) {
      // CHECK: %[[ARG0_SHAPE:.+]] = shapex.make_ranked_shape %arg2 : (index) -> !shapex.ranked_shape<[?,4]>
      %arg0_shape = shapex.make_ranked_shape %arg0_dim0 : (index) -> !shapex.ranked_shape<[?,4]>
      // CHECK: %[[ARG0_DIM0:.+]] = shapex.ranked_dim %[[ARG0_SHAPE]][0] : !shapex.ranked_shape<[?,4]> -> index
      // CHECK: %[[ARG0_SPAN:.+]] = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:?x4xf32>{%[[ARG0_DIM0]]}
      // CHECK: = flow.dispatch.tie_shape %[[ARG0_SPAN]], %[[ARG0_SHAPE]] : (!flow.dispatch.tensor<readonly:?x4xf32>, !shapex.ranked_shape<[?,4]>) -> !flow.dispatch.tensor<readonly:?x4xf32>
      %arg0_shaped = flow.dispatch.tie_shape %arg0, %arg0_shape : (!flow.dispatch.tensor<readonly:?x4xf32>, !shapex.ranked_shape<[?,4]>) -> !flow.dispatch.tensor<readonly:?x4xf32>

      // CHECK: %[[ARG1_SHAPE:.+]] = shapex.make_ranked_shape %arg3 : (index) -> !shapex.ranked_shape<[?]>
      %arg1_shape = shapex.make_ranked_shape %arg1_dim0 : (index) -> !shapex.ranked_shape<[?]>
      // CHECK: %[[ARG1_DIM0:.+]] = shapex.ranked_dim %[[ARG1_SHAPE]][0] : !shapex.ranked_shape<[?]> -> index
      // CHECK: %[[ARG1_SPAN:.+]] = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:?xf32>{%[[ARG1_DIM0]]}
      // CHECK: = flow.dispatch.tie_shape %[[ARG1_SPAN]], %[[ARG1_SHAPE]] : (!flow.dispatch.tensor<writeonly:?xf32>, !shapex.ranked_shape<[?]>) -> !flow.dispatch.tensor<writeonly:?xf32>
      %arg1_shaped = flow.dispatch.tie_shape %arg1, %arg1_shape : (!flow.dispatch.tensor<writeonly:?xf32>, !shapex.ranked_shape<[?]>) -> !flow.dispatch.tensor<writeonly:?xf32>

      return
    }
  }
}

// CHECK-LABEL: @simple_mul
func @simple_mul(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: %[[DIM0:.+]] = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %dim0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  // CHECK: %[[ARG0_SIZE:.+]] = stream.tensor.sizeof tensor<?x4xf32>{%[[DIM0]]} : index
  // CHECK: %[[ARG0_IMPORT:.+]] = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x4xf32>{%[[DIM0]]} in !stream.resource<external>{%[[ARG0_SIZE]]}
  // CHECK: %[[ARG0_T:.+]] = stream.async.transfer %[[ARG0_IMPORT]] : !stream.resource<external>{%[[ARG0_SIZE]]} -> !stream.resource<*>{%[[ARG0_SIZE]]}
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<?x4xf32>{%dim0}

  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[RET0_SIZE:.+]] = stream.tensor.sizeof tensor<?xf32>{%[[DIM0]]} : index
  // CHECK: %[[RET0:.+]] = stream.async.dispatch @executable::@dispatch[%c4, %c1, %c1](%[[ARG0_T]]) : (!stream.resource<*>{%[[ARG0_SIZE]]}) -> !stream.resource<*>{%[[RET0_SIZE]]}
  %1 = flow.dispatch @executable::@dispatch[%c4, %c1, %c1](%0) : (tensor<?x4xf32>{%dim0}) -> tensor<?xf32>{%dim0}

  // CHECK: %[[RET0_T:.+]] = stream.async.transfer %[[RET0]] : !stream.resource<*>{%[[RET0_SIZE]]} -> !stream.resource<external>{%[[RET0_SIZE]]}
  // CHECK: %[[RET0_EXPORT:.+]] = stream.tensor.export %[[RET0_T]] : tensor<?xf32>{%[[DIM0]]} in !stream.resource<external>{%[[RET0_SIZE]]} -> !hal.buffer_view
  %2 = hal.tensor.cast %1 : tensor<?xf32>{%dim0} -> !hal.buffer_view
  // CHECK: return %[[RET0_EXPORT]] : !hal.buffer_view
  return %2 : !hal.buffer_view
}
