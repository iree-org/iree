// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @optimizationBarrier
util.func public @optimizationBarrier(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-SAME: %[[ARG0:.+]]: !stream.resource<*>
  // CHECK: %[[RESOURCE:.*]] = util.optimization_barrier %[[ARG0]]
  // CHECK: %[[SIZE:.*]] = stream.resource.size %[[RESOURCE]] : !stream.resource<*>
  // CHECK: util.return %[[RESOURCE]], %[[SIZE]] : !stream.resource<*>, index
  %0 = util.optimization_barrier %arg0 : tensor<i32>
  util.return %0 : tensor<i32>
}
