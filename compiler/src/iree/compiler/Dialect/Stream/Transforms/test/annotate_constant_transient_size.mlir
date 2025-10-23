// RUN: iree-opt --split-input-file --iree-stream-annotate-constant-transient-size %s | FileCheck %s

// Test with no transients (pass should be no-op).
// CHECK-LABEL: @no_transients
util.func public @no_transients(%arg0: !stream.resource<*>, %arg0_size: index) -> (!stream.resource<*>, index) {
  // CHECK-NOT: iree.reflection
  // CHECK: util.return
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// -----

// Test with constant size query function.
// CHECK-LABEL: @constant_size_query
util.func public @constant_size_query(%arg0: !stream.resource<*>) -> (!stream.resource<*>) {
  // TODO(benvanik): Add iree.reflection metadata pointing to size query.
  // CHECK: util.return
  util.return %arg0 : !stream.resource<*>
}

// Size query function that folds to constant.
util.func private @constant_size_query$transient_size() -> index {
  %c1024 = arith.constant 1024 : index
  // TODO(benvanik): Verify pass detects constant return.
  // TODO(benvanik): Verify iree.reflection metadata added with constant value.
  util.return %c1024 : index
}

// -----

// Test with dynamic size query function (should not annotate).
// CHECK-LABEL: @dynamic_size_query
util.func public @dynamic_size_query(%arg0: !stream.resource<*>, %dim: index) -> (!stream.resource<*>) {
  // TODO(benvanik): Add iree.reflection metadata pointing to size query.
  // CHECK: util.return
  util.return %arg0 : !stream.resource<*>
}

// Size query function with dynamic computation.
util.func private @dynamic_size_query$transient_size(%dim: index) -> index {
  %c16 = arith.constant 16 : index
  %size = arith.muli %dim, %c16 : index
  // TODO(benvanik): Verify pass does NOT annotate (dynamic computation).
  util.return %size : index
}
