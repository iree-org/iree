// RUN: iree-opt --split-input-file --iree-stablehlo-inline-shape-assertion-helpers %s | FileCheck %s

// Nested private helpers feeding an assertion are inlined bottom-up.
// CHECK-LABEL: func.func public @nested_helpers
// CHECK-NOT: call
// CHECK: %[[SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 1
// CHECK: %[[REM:.+]] = stablehlo.remainder %[[SIZE]]
// CHECK: %[[QUOT:.+]] = stablehlo.divide %[[SIZE]]
// CHECK-NOT: call
// CHECK: stablehlo.custom_call @shape_assertion
func.func public @nested_helpers(%arg0: tensor<3x?x5xf32>) {
  %two = stablehlo.constant dense<2> : tensor<i32>
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?x5xf32>) -> tensor<i32>
  %qr:2 = call @divmod(%size, %two) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %even = stablehlo.compare EQ, %qr#1, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%even, %qr#0) {error_message = "odd", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  return
}
func.func private @divmod(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %r = call @rem(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %q = call @quot(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %q, %r : tensor<i32>, tensor<i32>
}
func.func private @rem(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.remainder %x, %y : tensor<i32>
  return %0 : tensor<i32>
}
func.func private @quot(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.divide %x, %y : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// A call marked no_inline stays.
// CHECK-LABEL: func.func public @no_inline_call
// CHECK: call @rem
func.func public @no_inline_call(%arg0: tensor<4x?xf32>) {
  %three = stablehlo.constant dense<3> : tensor<i32>
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<4x?xf32>) -> tensor<i32>
  %r = call @rem(%size, %three) <{no_inline}> : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %ok = stablehlo.compare EQ, %r, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "rem", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @rem(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.remainder %x, %y : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// A placed helper stays.
// CHECK-LABEL: func.func public @placed_helper
// CHECK: call @rem
func.func public @placed_helper(%arg0: tensor<4x?xf32>) {
  %three = stablehlo.constant dense<3> : tensor<i32>
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<4x?xf32>) -> tensor<i32>
  %r = call @rem(%size, %three) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %ok = stablehlo.compare EQ, %r, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "rem", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @rem(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> attributes {stream.affinity = #hal.device.promise<@dev>} {
  %0 = stablehlo.remainder %x, %y : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// A helper with a side effect stays.
// CHECK-LABEL: func.func public @effectful_helper
// CHECK: call @noisy
func.func public @effectful_helper(%arg0: tensor<4x?xf32>) {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<4x?xf32>) -> tensor<i32>
  %r = call @noisy(%size) : (tensor<i32>) -> tensor<i32>
  %ok = stablehlo.compare GE, %r, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "neg", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @noisy(%x: tensor<i32>) -> tensor<i32> {
  stablehlo.custom_call @log(%x) {has_side_effect = true} : (tensor<i32>) -> ()
  return %x : tensor<i32>
}

// -----

// A helper over non-scalar tensors stays.
// CHECK-LABEL: func.func public @tensor_helper
// CHECK: call @sum
func.func public @tensor_helper(%arg0: tensor<3x4xi32>) {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %s = call @sum(%arg0) : (tensor<3x4xi32>) -> tensor<i32>
  %ok = stablehlo.compare GE, %s, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "neg", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @sum(%x: tensor<3x4xi32>) -> tensor<i32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x4xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// A recursive helper stays.
// CHECK-LABEL: func.func public @recursive_helper
// CHECK: call @down
func.func public @recursive_helper(%arg0: tensor<4x?xf32>) {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<4x?xf32>) -> tensor<i32>
  %r = call @down(%size) : (tensor<i32>) -> tensor<i32>
  %ok = stablehlo.compare GE, %r, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "neg", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @down(%x: tensor<i32>) -> tensor<i32> {
  %0 = call @down(%x) : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// A scalar helper that feeds no assertion stays.
// CHECK-LABEL: func.func public @unrelated_helper
// CHECK: call @rem
func.func public @unrelated_helper(%arg0: tensor<4x?xf32>) -> tensor<i32> {
  %three = stablehlo.constant dense<3> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<4x?xf32>) -> tensor<i32>
  %r = call @rem(%size, %three) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %r : tensor<i32>
}
func.func private @rem(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.remainder %x, %y : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// A call in a nested module inlines the callee its own symbol table names.
// CHECK-LABEL: module @inner
// CHECK: func.func public @nested_table
// CHECK: stablehlo.remainder
// CHECK: stablehlo.custom_call @shape_assertion
module {
  func.func private @rem(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
    %zero = stablehlo.constant dense<0> : tensor<i32>
    return %zero : tensor<i32>
  }
  func.func public @use_outer(%a: tensor<i32>) -> tensor<i32> {
    %0 = call @rem(%a, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
  module @inner {
    func.func public @nested_table(%arg0: tensor<4x?xf32>) {
      %three = stablehlo.constant dense<3> : tensor<i32>
      %zero = stablehlo.constant dense<0> : tensor<i32>
      %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<4x?xf32>) -> tensor<i32>
      %r = call @rem(%size, %three) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %ok = stablehlo.compare EQ, %r, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.custom_call @shape_assertion(%ok) {error_message = "rem", has_side_effect = true} : (tensor<i1>) -> ()
      return
    }
    func.func private @rem(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
      %0 = stablehlo.remainder %x, %y : tensor<i32>
      return %0 : tensor<i32>
    }
  }
}

// -----

// CHECK-LABEL: func.func @boundary_call_noinline
// CHECK: call @helper
// CHECK: stablehlo.custom_call @shape_assertion
func.func @boundary_call_noinline(%x: i32) {
  %v = call @helper(%x) {noinline} : (i32) -> i32
  %zero = arith.constant 0 : i32
  %ok = arith.cmpi sge, %v, %zero : i32
  %pred = tensor.from_elements %ok : tensor<i1>
  stablehlo.custom_call @shape_assertion(%pred) {error_message = "nonnegative", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @helper(%x: i32) -> i32 {
  return %x : i32
}

// -----

// CHECK-LABEL: func.func @boundary_function_no_inline
// CHECK: call @helper
// CHECK: stablehlo.custom_call @shape_assertion
func.func @boundary_function_no_inline(%x: i32) {
  %v = call @helper(%x) : (i32) -> i32
  %zero = arith.constant 0 : i32
  %ok = arith.cmpi sge, %v, %zero : i32
  %pred = tensor.from_elements %ok : tensor<i1>
  stablehlo.custom_call @shape_assertion(%pred) {error_message = "nonnegative", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @helper(%x: i32) -> i32 attributes {no_inline} {
  return %x : i32
}

// -----

// CHECK-LABEL: func.func @boundary_function_noinline
// CHECK: call @helper
// CHECK: stablehlo.custom_call @shape_assertion
func.func @boundary_function_noinline(%x: i32) {
  %v = call @helper(%x) : (i32) -> i32
  %zero = arith.constant 0 : i32
  %ok = arith.cmpi sge, %v, %zero : i32
  %pred = tensor.from_elements %ok : tensor<i1>
  stablehlo.custom_call @shape_assertion(%pred) {error_message = "nonnegative", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @helper(%x: i32) -> i32 attributes {noinline} {
  return %x : i32
}

// -----

// CHECK-LABEL: func.func @boundary_placed_call
// CHECK: call @helper
// CHECK: stablehlo.custom_call @shape_assertion
func.func @boundary_placed_call(%x: i32) {
  %v = call @helper(%x) {stream.affinity = #hal.device.promise<@dev>} : (i32) -> i32
  %zero = arith.constant 0 : i32
  %ok = arith.cmpi sge, %v, %zero : i32
  %pred = tensor.from_elements %ok : tensor<i1>
  stablehlo.custom_call @shape_assertion(%pred) {error_message = "nonnegative", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @helper(%x: i32) -> i32 {
  return %x : i32
}

// -----

// A shared diamond counts both occurrences of each child, and inlines normally.
// CHECK-LABEL: func.func @diamond
// CHECK-NOT: call
// CHECK-COUNT-15: arith.addi
// CHECK-NOT: call
// CHECK: stablehlo.custom_call @shape_assertion
func.func @diamond(%x: i32) {
  %v = call @level2(%x) : (i32) -> i32
  %ok = arith.cmpi sge, %v, %x : i32
  %pred = tensor.from_elements %ok : tensor<i1>
  stablehlo.custom_call @shape_assertion(%pred) {error_message = "overflow", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @level0(%x: i32) -> i32 {
  %a = arith.addi %x, %x : i32
  %b = arith.addi %x, %x : i32
  %result = arith.addi %a, %b : i32
  return %result : i32
}
func.func private @level1(%x: i32) -> i32 {
  %a = call @level0(%x) : (i32) -> i32
  %b = call @level0(%x) : (i32) -> i32
  %result = arith.addi %a, %b : i32
  return %result : i32
}
func.func private @level2(%x: i32) -> i32 {
  %a = call @level1(%x) : (i32) -> i32
  %b = call @level1(%x) : (i32) -> i32
  %result = arith.addi %a, %b : i32
  return %result : i32
}

// -----

// Mutual recursion rejects the SCC and callers without changing their bodies.
// CHECK-LABEL: func.func @recursive_closure
// CHECK: call @first
// CHECK: stablehlo.custom_call @shape_assertion
// CHECK-LABEL: func.func private @first
// CHECK: call @second
// CHECK-LABEL: func.func private @second
// CHECK: call @first
func.func @recursive_closure(%x: i32) {
  %v = call @first(%x) : (i32) -> i32
  %ok = arith.cmpi eq, %v, %x : i32
  %pred = tensor.from_elements %ok : tensor<i1>
  stablehlo.custom_call @shape_assertion(%pred) {error_message = "recursive", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @first(%x: i32) -> i32 {
  %v = call @second(%x) : (i32) -> i32
  return %v : i32
}
func.func private @second(%x: i32) -> i32 {
  %v = call @first(%x) : (i32) -> i32
  return %v : i32
}

// -----

// A small repeated-call graph has exponential full expansion. Keep the root
// call and assertion when the expansion exceeds the pass-wide allowance.
// CHECK-LABEL: func.func @bounded_expansion
// CHECK: call @expand12
// CHECK: stablehlo.custom_call @shape_assertion
func.func @bounded_expansion(%x: i32) {
  %v = call @expand12(%x) : (i32) -> i32
  %ok = arith.cmpi eq, %v, %x : i32
  %pred = tensor.from_elements %ok : tensor<i1>
  stablehlo.custom_call @shape_assertion(%pred) {error_message = "expansion", has_side_effect = true} : (tensor<i1>) -> ()
  return
}
func.func private @expand0(%x: i32) -> i32 {
  %v = arith.addi %x, %x : i32
  return %v : i32
}
func.func private @expand1(%x: i32) -> i32 {
  %a = call @expand0(%x) : (i32) -> i32
  %b = call @expand0(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand2(%x: i32) -> i32 {
  %a = call @expand1(%x) : (i32) -> i32
  %b = call @expand1(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand3(%x: i32) -> i32 {
  %a = call @expand2(%x) : (i32) -> i32
  %b = call @expand2(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand4(%x: i32) -> i32 {
  %a = call @expand3(%x) : (i32) -> i32
  %b = call @expand3(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand5(%x: i32) -> i32 {
  %a = call @expand4(%x) : (i32) -> i32
  %b = call @expand4(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand6(%x: i32) -> i32 {
  %a = call @expand5(%x) : (i32) -> i32
  %b = call @expand5(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand7(%x: i32) -> i32 {
  %a = call @expand6(%x) : (i32) -> i32
  %b = call @expand6(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand8(%x: i32) -> i32 {
  %a = call @expand7(%x) : (i32) -> i32
  %b = call @expand7(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand9(%x: i32) -> i32 {
  %a = call @expand8(%x) : (i32) -> i32
  %b = call @expand8(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand10(%x: i32) -> i32 {
  %a = call @expand9(%x) : (i32) -> i32
  %b = call @expand9(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand11(%x: i32) -> i32 {
  %a = call @expand10(%x) : (i32) -> i32
  %b = call @expand10(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
func.func private @expand12(%x: i32) -> i32 {
  %a = call @expand11(%x) : (i32) -> i32
  %b = call @expand11(%x) : (i32) -> i32
  %v = arith.addi %a, %b : i32
  return %v : i32
}
