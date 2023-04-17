// RUN: iree-opt --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @ukernel_generic(
    %in0: tensor<?x?xf32>, %in1: tensor<?xf32>,
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>,
    %b0: f32, %b1: i64) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic "foo"
      ins(%in0, %in1: tensor<?x?xf32>, tensor<?xf32>)
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      (%b0, %b1 : f32, i64) -> tensor<?xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}
//      CHECK: func @ukernel_generic(
// CHECK-SAME:     %[[IN0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[IN1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: i64
//      CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic
// CHECK-SAME:       "foo"
// CHECK-SAME:       ins(%[[IN0]], %[[IN1]] :
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
// CHECK-SAME:       (%[[B0]], %[[B1]] :
//      CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @ukernel_generic_memref(
    %in0: memref<?x?xf32>, %in1: memref<?xf32>,
    %out0: memref<?xf32>, %out1 : memref<?x?xf32>,
    %b0: f32, %b1: i64) {
  iree_codegen.ukernel.generic "foo"
      ins(%in0, %in1: memref<?x?xf32>, memref<?xf32>)
      outs(%out0, %out1 : memref<?xf32>, memref<?x?xf32>)
      (%b0, %b1 : f32, i64)
  return
}
//      CHECK: func @ukernel_generic_memref(
// CHECK-SAME:     %[[IN0:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[IN1:[a-zA-Z0-9]+]]: memref<?xf32>
// CHECK-SAME:     %[[OUT0:[a-zA-Z0-9]+]]: memref<?xf32>
// CHECK-SAME:     %[[OUT1:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: i64
//      CHECK:   iree_codegen.ukernel.generic
// CHECK-SAME:       "foo"
// CHECK-SAME:       ins(%[[IN0]], %[[IN1]] :
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
// CHECK-SAME:       (%[[B0]], %[[B1]] :

// -----

func.func @ukernel_generic_optional_input(
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>,
    %b0: f32, %b1: i64) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic "foo"
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      (%b0, %b1 : f32, i64) -> tensor<?xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}
//      CHECK: func @ukernel_generic_optional_input(
//      CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic
//  CHECK-NOT:       ins

// -----

func.func @ukernel_generic_memref_optional_input(
    %out0: memref<?xf32>, %out1 : memref<?x?xf32>,
    %b0: f32, %b1: i64) {
  iree_codegen.ukernel.generic "foo"
      outs(%out0, %out1 : memref<?xf32>, memref<?x?xf32>)
      (%b0, %b1 : f32, i64)
  return
}
//      CHECK: func @ukernel_generic_memref_optional_input(
//      CHECK:   iree_codegen.ukernel.generic
//  CHECK-NOT:       ins

// -----

func.func @ukernel_generic_optional_other_operands(
    %in0: tensor<?x?xf32>, %in1: tensor<?xf32>,
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic "foo"
      ins(%in0, %in1: tensor<?x?xf32>, tensor<?xf32>)
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      -> tensor<?xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}
//      CHECK: func @ukernel_generic_optional_other_operands(
//      CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic
// CHECK-SAME:       outs(%{{.+}}, %{{.+}} : tensor<?xf32>, tensor<?x?xf32>) ->

// -----

func.func @ukernel_generic_non_tensor_memref_outs(
   %out0 : f32) -> f32 {
  // expected-error @+1 {{operand #0 must be ranked tensor of any type values or memref of any type values, but got 'f32'}}
  %0 = iree_codegen.ukernel.generic "foo"
      outs(%out0 : f32) -> f32
  return %0 : f32
}

// -----

func.func @ukernel_generic_err_tensor_outs(
    %out0: tensor<?xf32>, %out1 : memref<?x?xf32>) {
  // expected-error @+1 {{expected the number of results (0) to be equal to the number of output tensors (1)}}
  iree_codegen.ukernel.generic "foo"
      outs(%out0, %out1 : tensor<?xf32>, memref<?x?xf32>)
}

// -----

func.func @ukernel_generic_mixed_tensor_memref(
    %out0: tensor<?xf32>, %out1 : memref<?x?xf32>)
    -> tensor<?xf32> {
  %0 = iree_codegen.ukernel.generic "foo"
      outs(%out0, %out1 : tensor<?xf32>, memref<?x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//      CHECK: func @ukernel_generic_mixed_tensor_memref(
// CHECK-SAME:     %[[OUT0:.+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT1:.+]]: memref<?x?xf32>
//      CHECK:   %[[RESULT:.+]] = iree_codegen.ukernel.generic
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @ukernel_generic_err_memref_outs(
    %out0: tensor<?xf32>, %out1 : memref<?x?xf32>)
    -> (tensor<?xf32>, tensor<?x?xf32>){
  // expected-error @+1 {{expected the number of results (2) to be equal to the number of output tensors (1)}}
  %0:2 = iree_codegen.ukernel.generic "foo"
      outs(%out0, %out1 : tensor<?xf32>, memref<?x?xf32>) -> tensor<?xf32>, tensor<?x?xf32>
}

// -----

func.func @ukernel_generic_err_elementtype(
    %out0: tensor<?xf32>) -> tensor<?xi32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<?xf32>') to match type of corresponding result ('tensor<?xi32>')}}
  %0 = iree_codegen.ukernel.generic "foo"
      outs(%out0 : tensor<?xf32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

func.func @ukernel_generic_err_shape_mismatch(
    %out0: tensor<?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<?xf32>') to match type of corresponding result ('tensor<?x?xf32>')}}
  %0 = iree_codegen.ukernel.generic "foo"
      outs(%out0 : tensor<?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @ukernel_generic_err_static_shape_mismatch(
    %out0: tensor<10xf32>) -> tensor<20xf32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<10xf32>') to match type of corresponding result ('tensor<20xf32>')}}
  %0 = iree_codegen.ukernel.generic "foo"
      outs(%out0 : tensor<10xf32>) -> tensor<20xf32>
  return %0 : tensor<20xf32>
}

// -----

func.func @ukernel_generic_static_dynamic_shape_match(
    %out0: tensor<10xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<10xf32>') to match type of corresponding result ('tensor<?xf32>')}}
  %0 = iree_codegen.ukernel.generic "foo"
      outs(%out0 : tensor<10xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @ukernel_generic_raw(
    %in0: tensor<?x?xf32>, %in1: tensor<?xf32>,
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>,
    %b0: f32, %b1: i64) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic_raw "foo"
      ins(%in0, %in1: tensor<?x?xf32>, tensor<?xf32>)
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      (%b0, %b1 : f32, i64) -> tensor<?xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}
//      CHECK: func @ukernel_generic_raw(
// CHECK-SAME:     %[[IN0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[IN1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: i64
//      CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic_raw
// CHECK-SAME:       "foo"
// CHECK-SAME:       ins(%[[IN0]], %[[IN1]] :
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
// CHECK-SAME:       (%[[B0]], %[[B1]] :
//      CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @ukernel_generic_raw_memref(
    %in0: memref<?x?xf32>, %in1: memref<?xf32>,
    %out0: memref<?xf32>, %out1 : memref<?x?xf32>,
    %b0: f32, %b1: i64) {
  iree_codegen.ukernel.generic_raw "foo"
      ins(%in0, %in1: memref<?x?xf32>, memref<?xf32>)
      outs(%out0, %out1 : memref<?xf32>, memref<?x?xf32>)
      (%b0, %b1 : f32, i64)
  return
}
//      CHECK: func @ukernel_generic_raw_memref(
// CHECK-SAME:     %[[IN0:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[IN1:[a-zA-Z0-9]+]]: memref<?xf32>
// CHECK-SAME:     %[[OUT0:[a-zA-Z0-9]+]]: memref<?xf32>
// CHECK-SAME:     %[[OUT1:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: i64
//      CHECK:   iree_codegen.ukernel.generic_raw
// CHECK-SAME:       "foo"
// CHECK-SAME:       ins(%[[IN0]], %[[IN1]] :
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
// CHECK-SAME:       (%[[B0]], %[[B1]] :

// -----

func.func @ukernel_generic_raw_optional_input(
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>,
    %b0: f32, %b1: i64) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      (%b0, %b1 : f32, i64) -> tensor<?xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}
//      CHECK: func @ukernel_generic_raw_optional_input(
//      CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic_raw
//  CHECK-NOT:       ins

// -----

func.func @ukernel_generic_raw_memref_optional_input(
    %out0: memref<?xf32>, %out1 : memref<?x?xf32>,
    %b0: f32, %b1: i64) {
  iree_codegen.ukernel.generic_raw "foo"
      outs(%out0, %out1 : memref<?xf32>, memref<?x?xf32>)
      (%b0, %b1 : f32, i64)
  return
}
//      CHECK: func @ukernel_generic_raw_memref_optional_input(
//      CHECK:   iree_codegen.ukernel.generic_raw
//  CHECK-NOT:       ins

// -----

func.func @ukernel_generic_raw_optional_other_operands(
    %in0: tensor<?x?xf32>, %in1: tensor<?xf32>,
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic_raw "foo"
      ins(%in0, %in1: tensor<?x?xf32>, tensor<?xf32>)
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      -> tensor<?xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}
//      CHECK: func @ukernel_generic_raw_optional_other_operands(
//      CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic_raw
// CHECK-SAME:       outs(%{{.+}}, %{{.+}} : tensor<?xf32>, tensor<?x?xf32>) ->

// -----

func.func @ukernel_generic_raw_non_tensor_memref_outs(
   %out0 : f32) -> f32 {
  // expected-error @+1 {{operand #0 must be ranked tensor of any type values or memref of any type values, but got 'f32'}}
  %0 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0 : f32) -> f32
  return %0 : f32
}

// -----

func.func @ukernel_generic_raw_err_tensor_outs(
    %out0: tensor<?xf32>, %out1 : memref<?x?xf32>) {
  // expected-error @+1 {{expected the number of results (0) to be equal to the number of output tensors (1)}}
  iree_codegen.ukernel.generic_raw "foo"
      outs(%out0, %out1 : tensor<?xf32>, memref<?x?xf32>)
}

// -----

func.func @ukernel_generic_raw_mixed_tensor_memref(
    %out0: tensor<?xf32>, %out1 : memref<?x?xf32>)
    -> tensor<?xf32> {
  %0 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0, %out1 : tensor<?xf32>, memref<?x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//      CHECK: func @ukernel_generic_raw_mixed_tensor_memref(
// CHECK-SAME:     %[[OUT0:.+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT1:.+]]: memref<?x?xf32>
//      CHECK:   %[[RESULT:.+]] = iree_codegen.ukernel.generic_raw
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @ukernel_generic_raw_err_memref_outs(
    %out0: tensor<?xf32>, %out1 : memref<?x?xf32>)
    -> (tensor<?xf32>, tensor<?x?xf32>){
  // expected-error @+1 {{expected the number of results (2) to be equal to the number of output tensors (1)}}
  %0:2 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0, %out1 : tensor<?xf32>, memref<?x?xf32>) -> tensor<?xf32>, tensor<?x?xf32>
}

// -----

func.func @ukernel_generic_raw_err_elementtype(
    %out0: tensor<?xf32>) -> tensor<?xi32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<?xf32>') to match type of corresponding result ('tensor<?xi32>')}}
  %0 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0 : tensor<?xf32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

func.func @ukernel_generic_raw_err_shape_mismatch(
    %out0: tensor<?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<?xf32>') to match type of corresponding result ('tensor<?x?xf32>')}}
  %0 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0 : tensor<?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @ukernel_generic_raw_err_static_shape_mismatch(
    %out0: tensor<10xf32>) -> tensor<20xf32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<10xf32>') to match type of corresponding result ('tensor<20xf32>')}}
  %0 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0 : tensor<10xf32>) -> tensor<20xf32>
  return %0 : tensor<20xf32>
}

// -----

func.func @ukernel_generic_raw_static_dynamic_shape_match(
    %out0: tensor<10xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected type of operand #0 ('tensor<10xf32>') to match type of corresponding result ('tensor<?xf32>')}}
  %0 = iree_codegen.ukernel.generic_raw "foo"
      outs(%out0 : tensor<10xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @ukernel_mmt4d(%lhs : tensor<?x?x?x?xf32>,
    %rhs : tensor<?x?x?x?xf32>, %outs : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = iree_codegen.ukernel.mmt4d lhs(%lhs : tensor<?x?x?x?xf32>)
      rhs(%rhs : tensor<?x?x?x?xf32>) outs(%outs : tensor<?x?x?x?xf32>)
      accumulate(true) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: func @ukernel_mmt4d(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//      CHECK:   %[[RESULT:.+]] = iree_codegen.ukernel.mmt4d
// CHECK-SAME:       lhs(%[[LHS]] :
// CHECK-SAME:       rhs(%[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
// CHECK-SAME:       accumulate(true)
//      CHECK:   return %[[RESULT]]

// -----

func.func @ukernel_mmt4d_memref(%lhs : memref<?x?x?x?xf32>,
    %rhs : memref<?x?x?x?xf32>, %outs : memref<?x?x?x?xf32>) {
  iree_codegen.ukernel.mmt4d lhs(%lhs : memref<?x?x?x?xf32>)
      rhs(%rhs : memref<?x?x?x?xf32>) outs(%outs : memref<?x?x?x?xf32>)
      accumulate(false)
  return
}
//      CHECK: func @ukernel_mmt4d_memref(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
// CHECK-SAME:     %[[OUTS:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//      CHECK:   iree_codegen.ukernel.mmt4d
// CHECK-SAME:       lhs(%[[LHS]] :
// CHECK-SAME:       rhs(%[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
// CHECK-SAME:       accumulate(false)
