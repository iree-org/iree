// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @optimizationBarrier
func.func @optimizationBarrier(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: stream.async.transfer
  // CHECK: %[[RESOURCE:.*]] = util.optimization_barrier %0
  // CHECK: %[[SIZE:.*]] = stream.resource.size %1 : !stream.resource<*>
  // CHECK: return %[[RESOURCE]], %[[SIZE]] : !stream.resource<*>, index
  %0 = util.optimization_barrier %arg0 : tensor<i32>
  return %0 : tensor<i32>
}
