// RUN: iree-opt --split-input-file --iree-stream-conversion --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @functionExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//   CHECK-SAME: %[[ARG1:.+]]: i1,
//  CHECK-SAME:  %[[ARG2:.+]]: !stream.resource<*>, %[[ARG2_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index, i1, !stream.resource<*>, index)
util.func private @functionExpansion(%arg0: tensor<4x?xf32>, %arg1: i1, %arg2: tensor<i32>)
    -> (tensor<4x?xf32>, i1, tensor<i32>) {
  // CHECK-NEXT: %[[RET:.+]]:5 = util.call @callee(%[[ARG0]], %[[ARG0_SIZE]], %[[ARG1]], %[[ARG2]], %[[ARG2_SIZE]])
  // CHECK-SAME: : (!stream.resource<*>, index, i1, !stream.resource<*>, index) -> (!stream.resource<*>, index, i1, !stream.resource<*>, index)
  %0:3 = util.call @callee(%arg0, %arg1, %arg2) : (tensor<4x?xf32>, i1, tensor<i32>) -> (tensor<4x?xf32>, i1, tensor<i32>)
  // CHECK: util.return %[[RET]]#0, %[[RET]]#1, %[[RET]]#2,  %[[RET]]#3, %[[RET]]#4 : !stream.resource<*>, index, i1, !stream.resource<*>, index
  util.return %0#0, %0#1, %0#2 : tensor<4x?xf32>, i1, tensor<i32>
}

// CHECK: util.func private @callee
util.func private @callee(%arg0: tensor<4x?xf32>, %arg1: i1, %arg2: tensor<i32>)
    -> (tensor<4x?xf32>, i1, tensor<i32>) {
  util.return %arg0, %arg1, %arg2 : tensor<4x?xf32>, i1, tensor<i32>
}

// -----

// This is a regression test for https://github.com/iree-org/iree/issues/22974.
// The following should fail the conversion, but not give a segfault or any
// other unexpected error.

// expected-error @+1 {{failed to legalize operation 'util.func' that was explicitly marked illegal}}
util.func public @test(%arg0: !util.list<tensor<*xf32>>) {
    %c10 = arith.constant 10 : index
    %0 = util.list.get %arg0[%c10] : !util.list<tensor<*xf32>> -> tensor<?xf32>
    util.return
  }
