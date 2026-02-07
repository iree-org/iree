// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=64})' %s | FileCheck %s

// Verify the generated itoa builtin function body.
// CHECK:      vm.func private @__iree_string_itoa_u64(%arg0: i64) -> !vm.buffer {
// CHECK-DAG:    %[[ZERO_I64:.+]] = vm.const.i64.zero
// CHECK-DAG:    %[[C1_I32:.+]] = vm.const.i32 1
// CHECK-DAG:    %[[C1_I64:.+]] = vm.const.i64 1
// CHECK-DAG:    %[[C10:.+]] = vm.const.i64 10
// CHECK-DAG:    %[[C20:.+]] = vm.const.i64 20
// CHECK-DAG:    %[[C19:.+]] = vm.const.i64 19
// CHECK-DAG:    %[[ASCII_ZERO:.+]] = vm.const.i32 48
// CHECK:        %[[SCRATCH:.+]] = vm.buffer.alloc %[[C20]], %[[C1_I32]]
// CHECK:        %[[IS_ZERO:.+]] = vm.cmp.eq.i64 %arg0, %[[ZERO_I64]]
// CHECK:        vm.cond_br %[[IS_ZERO]], ^[[ZERO_BB:.+]], ^[[LOOP_BB:.+]](%arg0, %[[C19]] : i64, i64)
// Zero case: store '0', clone 1 byte, return.
// CHECK:      ^[[ZERO_BB]]:
// CHECK:        vm.buffer.store.i8 %[[ASCII_ZERO]], %[[SCRATCH]][%[[ZERO_I64]]]
// CHECK:        %[[ZERO_RESULT:.+]] = vm.buffer.clone %[[SCRATCH]], %[[ZERO_I64]], %[[C1_I64]], %[[C1_I32]]
// CHECK:        vm.return %[[ZERO_RESULT]]
// Loop: extract digit, store in reverse, divide by 10, loop until zero.
// CHECK:      ^[[LOOP_BB]](%[[REM:.+]]: i64, %[[POS:.+]]: i64):
// CHECK:        %[[DIGIT:.+]] = vm.rem.i64.u %[[REM]], %[[C10]]
// CHECK:        %[[DIGIT_I32:.+]] = vm.trunc.i64.i32 %[[DIGIT]]
// CHECK:        %[[ASCII:.+]] = vm.add.i32 %[[DIGIT_I32]], %[[ASCII_ZERO]]
// CHECK:        vm.buffer.store.i8 %[[ASCII]], %[[SCRATCH]][%[[POS]]]
// CHECK:        %[[NEXT_REM:.+]] = vm.div.i64.u %[[REM]], %[[C10]]
// CHECK:        %[[NEXT_POS:.+]] = vm.sub.i64 %[[POS]], %[[C1_I64]]
// CHECK:        %[[DONE:.+]] = vm.cmp.eq.i64 %[[NEXT_REM]], %[[ZERO_I64]]
// CHECK:        vm.cond_br %[[DONE]], ^[[EXTRACT_BB:.+]](%[[POS]] : i64), ^[[LOOP_BB]](%[[NEXT_REM]], %[[NEXT_POS]] : i64, i64)
// Extract: clone from start position to end of scratch buffer.
// CHECK:      ^[[EXTRACT_BB]](%[[START:.+]]: i64):
// CHECK:        %[[LENGTH:.+]] = vm.sub.i64 %[[C20]], %[[START]]
// CHECK:        %[[RESULT:.+]] = vm.buffer.clone %[[SCRATCH]], %[[START]], %[[LENGTH]], %[[C1_I32]]
// CHECK:        vm.return %[[RESULT]]
// CHECK:      }

// CHECK-LABEL: @string_itoa
func.func @string_itoa(%arg0 : index) -> !util.buffer {
  // CHECK: vm.call @__iree_string_itoa_u64(%arg0) : (i64) -> !vm.buffer
  %str = util.string.itoa %arg0 : index -> !util.buffer
  return %str : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_single_arg
func.func @string_format_single_arg(%arg0 : index) -> !util.buffer {
  // CHECK: vm.call @__iree_string_itoa_u64(%arg0) : (i64) -> !vm.buffer
  // CHECK: vm.buffer.length
  // CHECK: vm.add.i64
  // CHECK: vm.buffer.alloc
  // CHECK: vm.rodata.inline : !vm.buffer = "blk."
  // CHECK: vm.buffer.copy
  // CHECK: vm.buffer.copy
  // CHECK: vm.rodata.inline : !vm.buffer = ".attn_q.weight"
  // CHECK: vm.buffer.copy
  %key = util.string.format "blk.{}.attn_q.weight"(%arg0) : (index) -> !util.buffer
  return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_no_args
func.func @string_format_no_args() -> !util.buffer {
  // CHECK: vm.rodata.inline : !vm.buffer = "static_key"
  // CHECK-NOT: vm.buffer.copy
  %key = util.string.format "static_key"() : () -> !util.buffer
  return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_buffer_arg
func.func @string_format_buffer_arg(%scope : !util.buffer, %idx : index) -> !util.buffer {
  // CHECK: vm.call @__iree_string_itoa_u64
  // CHECK: vm.buffer.length
  // CHECK: vm.buffer.length
  // CHECK: vm.buffer.alloc
  // CHECK: vm.buffer.copy
  // CHECK: vm.buffer.copy
  // CHECK: vm.buffer.copy
  // CHECK: vm.buffer.copy
  %key = util.string.format "{}::blk.{}.weight"(%scope, %idx) : (!util.buffer, index) -> !util.buffer
  return %key : !util.buffer
}
