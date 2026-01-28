// RUN: iree-opt --split-input-file --iree-vm-convert-to-yieldable-calls %s | FileCheck %s

// Tests that vm.call to a function with vm.yield attribute is converted to
// vm.call.yieldable.

vm.module @module {
  // Import marked as yieldable.
  vm.import private @yieldable_fn(%arg: i32) -> i32 attributes {vm.yield}
  vm.import private @normal_fn(%arg: i32) -> i32

  // CHECK-LABEL: @call_yieldable_import
  vm.func @call_yieldable_import(%arg0: i32) -> i32 {
    // CHECK: vm.call.yieldable @yieldable_fn(%arg0) : (i32) -> ^bb1 (i32)
    // CHECK-NEXT: ^bb1(%[[RESULT:.*]]: i32):
    // CHECK-NEXT: vm.return %[[RESULT]] : i32
    %0 = vm.call @yieldable_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @call_normal_import
  vm.func @call_normal_import(%arg0: i32) -> i32 {
    // CHECK: %[[RESULT:.*]] = vm.call @normal_fn(%arg0) : (i32) -> i32
    // CHECK-NEXT: vm.return %[[RESULT]] : i32
    %0 = vm.call @normal_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// Tests that internal functions with vm.yield attribute are converted.

vm.module @module_internal {
  // Internal function marked as yieldable.
  vm.func private @yieldable_internal(%arg: i32) -> i32 attributes {vm.yield} {
    %c1 = vm.const.i32 1
    %result = vm.add.i32 %arg, %c1 : i32
    vm.return %result : i32
  }

  // CHECK-LABEL: @call_yieldable_internal
  vm.func @call_yieldable_internal(%arg0: i32) -> i32 {
    // CHECK: vm.call.yieldable @yieldable_internal(%arg0) : (i32) -> ^bb1 (i32)
    // CHECK-NEXT: ^bb1(%[[RESULT:.*]]: i32):
    // CHECK-NEXT: vm.return %[[RESULT]] : i32
    %0 = vm.call @yieldable_internal(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// Tests conversion with multiple results.

vm.module @module_multi_result {
  vm.import private @yieldable_multi(%arg: i32) -> (i32, i32) attributes {vm.yield}

  // CHECK-LABEL: @call_yieldable_multi_result
  vm.func @call_yieldable_multi_result(%arg0: i32) -> (i32, i32) {
    // CHECK: vm.call.yieldable @yieldable_multi(%arg0) : (i32) -> ^bb1 (i32, i32)
    // CHECK-NEXT: ^bb1(%[[R0:.*]]: i32, %[[R1:.*]]: i32):
    // CHECK-NEXT: vm.return %[[R0]], %[[R1]] : i32, i32
    %0:2 = vm.call @yieldable_multi(%arg0) : (i32) -> (i32, i32)
    vm.return %0#0, %0#1 : i32, i32
  }
}

// -----

// Tests conversion with ref types.

vm.module @module_ref {
  vm.import private @yieldable_ref(%buf: !vm.buffer) -> !vm.buffer attributes {vm.yield}

  // CHECK-LABEL: @call_yieldable_ref
  vm.func @call_yieldable_ref(%arg0: !vm.buffer) -> !vm.buffer {
    // CHECK: vm.call.yieldable @yieldable_ref(%arg0) : (!vm.buffer) -> ^bb1 (!vm.buffer)
    // CHECK-NEXT: ^bb1(%[[RESULT:.*]]: !vm.buffer):
    // CHECK-NEXT: vm.return %[[RESULT]] : !vm.buffer
    %0 = vm.call @yieldable_ref(%arg0) : (!vm.buffer) -> !vm.buffer
    vm.return %0 : !vm.buffer
  }
}

// -----

// Tests that code after the call is preserved in the resume block.

vm.module @module_code_after {
  vm.import private @yieldable_fn(%arg: i32) -> i32 attributes {vm.yield}

  // CHECK-LABEL: @call_with_code_after
  vm.func @call_with_code_after(%arg0: i32) -> i32 {
    // CHECK: vm.call.yieldable @yieldable_fn(%arg0) : (i32) -> ^bb1 (i32)
    // CHECK-NEXT: ^bb1(%[[R:.*]]: i32):
    // CHECK: %[[ADD:.*]] = vm.add.i32 %[[R]], %[[R]]
    // CHECK-NEXT: vm.return %[[ADD]] : i32
    %0 = vm.call @yieldable_fn(%arg0) : (i32) -> i32
    %1 = vm.add.i32 %0, %0 : i32
    vm.return %1 : i32
  }
}

// -----

// Tests multiple yieldable calls in sequence.

vm.module @module_sequence {
  vm.import private @yieldable_fn(%arg: i32) -> i32 attributes {vm.yield}

  // CHECK-LABEL: @call_sequence
  vm.func @call_sequence(%arg0: i32) -> i32 {
    // CHECK: vm.call.yieldable @yieldable_fn(%arg0) : (i32) -> ^bb1 (i32)
    // CHECK-NEXT: ^bb1(%[[R1:.*]]: i32):
    // CHECK: vm.call.yieldable @yieldable_fn(%[[R1]]) : (i32) -> ^bb2 (i32)
    // CHECK-NEXT: ^bb2(%[[R2:.*]]: i32):
    // CHECK-NEXT: vm.return %[[R2]] : i32
    %0 = vm.call @yieldable_fn(%arg0) : (i32) -> i32
    %1 = vm.call @yieldable_fn(%0) : (i32) -> i32
    vm.return %1 : i32
  }
}

// -----

// Tests that vm.call.variadic to a yieldable import is converted to
// vm.call.variadic.yieldable.

vm.module @module_variadic {
  vm.import private @yieldable_variadic(%arg: i32 ...) -> i32 attributes {vm.yield}
  vm.import private @normal_variadic(%arg: i32 ...) -> i32

  // CHECK-LABEL: @call_yieldable_variadic
  vm.func @call_yieldable_variadic(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: vm.call.variadic.yieldable @yieldable_variadic(%arg0, %arg1)
    // CHECK-SAME: {segment_sizes = dense<2> : vector<1xi16>, segment_types = [i32]}
    // CHECK-SAME: : (i32, i32) -> ^bb1 (i32)
    // CHECK-NEXT: ^bb1(%[[RESULT:.*]]: i32):
    // CHECK-NEXT: vm.return %[[RESULT]] : i32
    %0 = vm.call.variadic @yieldable_variadic([%arg0, %arg1]) : (i32 ...) -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @call_normal_variadic
  vm.func @call_normal_variadic(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: %[[RESULT:.*]] = vm.call.variadic @normal_variadic([%arg0, %arg1])
    // CHECK-SAME: : (i32 ...) -> i32
    // CHECK-NEXT: vm.return %[[RESULT]] : i32
    %0 = vm.call.variadic @normal_variadic([%arg0, %arg1]) : (i32 ...) -> i32
    vm.return %0 : i32
  }
}
