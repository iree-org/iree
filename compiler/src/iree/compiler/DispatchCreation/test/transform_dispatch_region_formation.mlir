// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: util.func public @single_op(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
util.func public @single_op(%arg0: tensor<?x?xf32>, %s1: index, %s2: index) -> tensor<?x?xf32> {
  // CHECK: %[[region:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%[[s1]], %[[s2]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   flow.return %[[slice]]
  // CHECK: }
  // CHECK: util.return %[[region]]
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.extract_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.wrap_in_dispatch_region %0 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// CHECK-LABEL: util.func public @clone_preceding(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
util.func public @clone_preceding(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[dim0:.*]] = tensor.dim %[[arg1]], %[[c0]]
  // CHECK-DAG: %[[dim1:.*]] = tensor.dim %[[arg1]], %[[c1]]
  // CHECK: %[[dummy:.*]] = "test.dummy"
  // CHECK: %[[region:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%[[dim0]], %[[dim1]]}) {
  // CHECK:   %[[dummy_clone:.*]] = "test.dummy"
  // CHECK:   %[[insert:.*]] = tensor.insert_slice %[[dummy_clone]] into %[[arg1]]
  // CHECK:   flow.return %[[insert]]
  // CHECK: }
  // CHECK: util.return %[[dummy]], %[[region]]
  %0 = "test.dummy"() : () -> (tensor<?x?xf32>)
  %1 = tensor.insert_slice %0 into %arg1 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.dummy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.clone_preceding_op_into_dispatch_region %1 into %dispatch_op : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// CHECK-LABEL: util.func public @move_preceding(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
util.func public @move_preceding(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[dim0:.*]] = tensor.dim %[[arg1]], %[[c0]]
  // CHECK-DAG: %[[dim1:.*]] = tensor.dim %[[arg1]], %[[c1]]
  // CHECK: %[[region:.*]]:2 = flow.dispatch.region -> (tensor<?x?xf32>{%[[dim0]], %[[dim1]]}, tensor<?x?xf32>{%[[s1]], %[[s2]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   %[[insert:.*]] = tensor.insert_slice %[[slice]] into %[[arg1]]
  // CHECK:   flow.return %[[insert]], %[[slice]]
  // CHECK: }
  // CHECK: util.return %[[region]]#0, %[[region]]#1
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %1, %0 : tensor<?x?xf32>, tensor<?x?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.extract_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.move_preceding_op_into_dispatch_region %1 into %dispatch_op : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// CHECK-LABEL: util.func public @create_region_and_convert_to_workgroups
//       CHECK:   tensor.empty()
//       CHECK:   flow.dispatch.workgroups
//       CHECK:     linalg.matmul
//       CHECK:     flow.return
util.func public @create_region_and_convert_to_workgroups(
    %A: tensor<5x3xf32>, %B: tensor<3x5xf32>) -> tensor<5x5xf32> {
  %init = tensor.empty() : tensor<5x5xf32>
  %matmul = linalg.matmul
      ins(%A, %B : tensor<5x3xf32>, tensor<3x5xf32>)
      outs(%init : tensor<5x5xf32>) -> tensor<5x5xf32>
  util.return %matmul : tensor<5x5xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %region_op = transform.iree.wrap_in_dispatch_region %0 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op
    transform.iree.region_to_workgroups %region_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// CHECK-LABEL: util.func public @clone_multiple_preceding
//   CHECK-DAG:   arith.constant
//   CHECK-DAG:   arith.constant
//   CHECK-DAG:   tensor.dim
//   CHECK-DAG:   tensor.dim
//       CHECK:   flow.dispatch.region
//  CHECK-NEXT:     "test.dummy_op"
//  CHECK-NEXT:     "test.first_user"
//  CHECK-NEXT:     "test.second_user"
//  CHECK-NEXT:     "test.merge1"
//  CHECK-NEXT:     "test.merge2"
util.func public @clone_multiple_preceding(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> (tensor<?x?xf32>) {
  %0 = "test.dummy_op"(%arg0) {__tagged__} : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
  %1 = "test.first_user"(%0) {__tagged__} : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
  %2 = "test.second_user"(%0) {__tagged__} : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
  %u = "test.third_user"(%0) : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
  %3 = "test.merge1"(%1, %2) {__tagged__} : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
  %4 = "test.merge2"(%1, %3) {__tagged__} : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
  %5 = tensor.insert_slice %4 into %arg1 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %5 : tensor<?x?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes{"__tagged__"} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.clone_preceding_op_into_dispatch_region %1 into %dispatch_op : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// CHECK-LABEL: util.func public @move_succeeding(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
util.func public @move_succeeding(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[dim0:.*]] = tensor.dim %[[arg1]], %[[c0]]
  // CHECK-DAG: %[[dim1:.*]] = tensor.dim %[[arg1]], %[[c1]]
  // CHECK: %[[region:.*]]:2 = flow.dispatch.region -> (tensor<?x?xf32>{%[[s1]], %[[s2]]}, tensor<?x?xf32>{%[[dim0]], %[[dim1]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   %[[insert:.*]] = tensor.insert_slice %[[slice]] into %[[arg1]]
  // CHECK:   flow.return %[[slice]], %[[insert]]
  // CHECK: }
  // CHECK: util.return %[[region]]#1, %[[region]]#0
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %1, %0 : tensor<?x?xf32>, tensor<?x?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.extract_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.move_succeeding_op_into_dispatch_region %1 into %dispatch_op : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// CHECK-LABEL: util.func public @move_multiple_succeeding
//  CHECK-NEXT:   flow.dispatch.region -> (tensor<50x90xf32>, tensor<50x90xf32>, tensor<50x90xf32>, tensor<50x90xf32>, tensor<50x90xf32>, tensor<600x700xf32>)
//  CHECK-NEXT:   "test.dummy_op"
//  CHECK-NEXT:   "test.first_user"
//  CHECK-NEXT:   "test.second_user"
//  CHECK-NEXT:   "test.merge1"
//  CHECK-NEXT:   "test.merge2"
//  CHECK-NEXT:   tensor.insert_slice
//  CHECK-NEXT:   flow.return
//  CHECK-NEXT: }
//  CHECK-NEXT: "test.third_user"
util.func public @move_multiple_succeeding(%arg0: tensor<50x90xf32>, %arg1: tensor<600x700xf32>) -> (tensor<600x700xf32>, tensor<50x90xf32>) {
  %0 = "test.dummy_op"(%arg0) : (tensor<50x90xf32>) -> (tensor<50x90xf32>)
  %1 = "test.first_user"(%0) {__tagged__} : (tensor<50x90xf32>) -> (tensor<50x90xf32>)
  %2 = "test.second_user"(%0) {__tagged__} : (tensor<50x90xf32>) -> (tensor<50x90xf32>)
  %u = "test.third_user"(%0) : (tensor<50x90xf32>) -> (tensor<50x90xf32>)
  %3 = "test.merge1"(%1, %2) {__tagged__} : (tensor<50x90xf32>, tensor<50x90xf32>) -> (tensor<50x90xf32>)
  %4 = "test.merge2"(%1, %3) {__tagged__} : (tensor<50x90xf32>, tensor<50x90xf32>) -> (tensor<50x90xf32>)
  %5 = tensor.insert_slice %4 into %arg1 [5, 16] [50, 90] [1, 1] {__tagged__}
      : tensor<50x90xf32> into tensor<600x700xf32>
  util.return %5, %u : tensor<600x700xf32>, tensor<50x90xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["test.dummy_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0  {generateWorkload=false} : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes{"__tagged__"} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.move_succeeding_op_into_dispatch_region %1 into %dispatch_op : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// CHECK-LABEL: util.func public @clone_succeeding(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
util.func public @clone_succeeding(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK: %[[region:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%[[s1]], %[[s2]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   tensor.insert_slice %[[slice]] into %[[arg1]]
  // CHECK:   flow.return %[[slice]]
  // CHECK: }
  // CHECK: %[[insert:.*]] = tensor.insert_slice %[[region]] into %[[arg1]]
  // CHECK: util.return %[[insert]], %[[region]]
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %1, %0 : tensor<?x?xf32>, tensor<?x?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.extract_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0  {generateWorkload=false} : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.clone_succeeding_op_into_dispatch_region %1 into %dispatch_op : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// This is a regression for reifyDynamicResultDims.

// CHECK-LABEL: util.func public @reify_result_dims_regression(
util.func public @reify_result_dims_regression(%s1: index, %s2: index) -> (tensor<4x?xf32>) {
  // CHECK: %[[dest:.*]] = "test.dummy_dest"
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: %[[dim1:.*]] = tensor.dim %[[dest]], %[[c1]]
  // CHECK: %[[src:.*]] = "test.dummy_src"
  // CHECK: %[[region:.*]] = flow.dispatch.region -> (tensor<4x?xf32>{%[[dim1]]}) {
  // CHECK:   %[[insert:.*]] = tensor.insert_slice %[[src]] into %[[dest]]
  // CHECK:   flow.return %[[insert]]
  // CHECK: }
  // CHECK: util.return %[[region]]

  // This op does not implement any interface for querying dynamic result dims.
  // Generate a tensor.dim op.
  %dest = "test.dummy_dest"() : () -> (tensor<4x?xf32>)
  %src = "test.dummy_src"() : () -> (tensor<?x?xf32>)
  %1 = tensor.insert_slice %src into %dest [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<4x?xf32>
  util.return %1 : tensor<4x?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module
