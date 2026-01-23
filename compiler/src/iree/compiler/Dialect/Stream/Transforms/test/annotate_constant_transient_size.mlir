// RUN: iree-opt --split-input-file --verify-diagnostics --iree-stream-annotate-constant-transient-size %s | FileCheck %s

// Tests that functions without transients are skipped.

// CHECK-LABEL: @no_transients
// CHECK-NOT: iree.reflection
util.func public @no_transients(%arg0: !stream.resource<*>, %arg0_size: index) -> (!stream.resource<*>, index) {
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests that a constant transient size is annotated in reflection metadata.

// CHECK-LABEL: @constant_size
// CHECK-SAME: iree.reflection = {iree.abi.transients.size = @constant_size$transient_size, iree.abi.transients.size.constant = 1024 : index}
util.func public @constant_size(%arg0: !stream.resource<*>) -> !stream.resource<*>
    attributes {iree.reflection = {iree.abi.transients.size = @constant_size$transient_size}} {
  util.return %arg0 : !stream.resource<*>
}

util.func private @constant_size$transient_size(%arg0: !stream.resource<*>) -> index {
  %c1024 = arith.constant 1024 : index
  util.return %c1024 : index
}

// -----

// Tests that existing constant annotations are preserved (user override).

// CHECK-LABEL: @already_annotated
// CHECK-SAME: iree.abi.transients.size.constant = 2048 : index
util.func public @already_annotated(%arg0: !stream.resource<*>) -> !stream.resource<*>
    attributes {iree.reflection = {
      iree.abi.transients.size = @already_annotated$transient_size,
      iree.abi.transients.size.constant = 2048 : index
    }} {
  util.return %arg0 : !stream.resource<*>
}

util.func private @already_annotated$transient_size(%arg0: !stream.resource<*>) -> index {
  %c1024 = arith.constant 1024 : index
  util.return %c1024 : index
}

// -----

// Tests that non-constant computations are skipped silently.

// CHECK-LABEL: @non_constant
// CHECK-SAME: iree.abi.transients.size = @non_constant$transient_size
// CHECK-NOT: iree.abi.transients.size.constant
util.func public @non_constant(%arg0: !stream.resource<*>) -> !stream.resource<*>
    attributes {iree.reflection = {iree.abi.transients.size = @non_constant$transient_size}} {
  util.return %arg0 : !stream.resource<*>
}

util.func private @non_constant$transient_size(%arg0: !stream.resource<*>) -> index {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %size = arith.addi %c16, %c32 : index
  util.return %size : index
}

// -----

// Verifies an error is emitted when the referenced size query function is not found.

// expected-error @+1 {{referenced transient size query function '@missing_func' not found in module}}
util.func public @symbol_not_found(%arg0: !stream.resource<*>) -> !stream.resource<*>
    attributes {iree.reflection = {iree.abi.transients.size = @missing_func}} {
  util.return %arg0 : !stream.resource<*>
}

// -----

// Verifies an error is emitted when the size query function has the wrong return type.

util.func public @wrong_type(%arg0: !stream.resource<*>) -> !stream.resource<*>
    attributes {iree.reflection = {iree.abi.transients.size = @wrong_type$transient_size}} {
  util.return %arg0 : !stream.resource<*>
}

// expected-error @+1 {{transient size query function must return index type}}
util.func private @wrong_type$transient_size(%arg0: !stream.resource<*>) -> i64 {
  %c1024 = arith.constant 1024 : i64
  util.return %c1024 : i64
}

// -----

// Verifies an error is emitted when the size query function has no return values.

util.func public @no_return(%arg0: !stream.resource<*>) -> !stream.resource<*>
    attributes {iree.reflection = {iree.abi.transients.size = @no_return$transient_size}} {
  util.return %arg0 : !stream.resource<*>
}

// expected-error @+1 {{transient size query function must return exactly one value}}
util.func private @no_return$transient_size(%arg0: !stream.resource<*>) {
  util.return
}

// -----

// Verifies a warning is emitted when the size query function returns multiple values.

util.func public @multi_value(%arg0: !stream.resource<*>) -> !stream.resource<*>
    attributes {iree.reflection = {iree.abi.transients.size = @multi_value$transient_size}} {
  util.return %arg0 : !stream.resource<*>
}

// expected-warning @+1 {{transient size query with multiple return values not yet supported for constant annotation}}
util.func private @multi_value$transient_size(%arg0: !stream.resource<*>) -> (index, index) {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  util.return %c128, %c256 : index, index
}
