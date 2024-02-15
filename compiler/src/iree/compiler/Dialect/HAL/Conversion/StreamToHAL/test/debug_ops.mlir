// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: @tensorTrace
// CHECK-SAME: (%[[TENSOR0_BUFFER:.+]]: !hal.buffer, %[[TENSOR0_SIZE:.+]]: index, %[[TENSOR1_BUFFER:.+]]: !hal.buffer, %[[TENSOR1_SIZE:.+]]: index, %[[TENSOR1_DIM0:.+]]: index)
util.func public @tensorTrace(%tensor0: !stream.resource<staging>, %tensor0_size: index, %tensor1: !stream.resource<staging>, %tensor1_size: index, %tensor1_dim0: index) {
  // CHECK-DAG: %[[TENSOR0:.+]] = hal.buffer_view.create buffer(%[[TENSOR0_BUFFER]] : !hal.buffer)[%c0{{.*}}, %[[TENSOR0_SIZE]]] shape([%c5, %c3])
  // CHECK-DAG: %[[TENSOR1:.+]] =  hal.buffer_view.create buffer(%[[TENSOR1_BUFFER]] : !hal.buffer)[%c0{{.*}}, %[[TENSOR1_SIZE]]] shape([%[[TENSOR1_DIM0]], %c5{{.*}}])
  // CHECK: hal.buffer_view.trace "FOOBAR" = %[[TENSOR0]], %[[TENSOR1]] : !hal.buffer_view, !hal.buffer_view
  stream.tensor.trace "FOOBAR" = [
    %tensor0 : tensor<5x3xf32> in !stream.resource<staging>{%tensor0_size},
    %tensor1 : tensor<?x5xf32>{%tensor1_dim0} in !stream.resource<staging>{%tensor1_size}
  ]
  util.return
}
