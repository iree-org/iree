// RUN: iree-opt --split-input-file --iree-vm-conversion %s | FileCheck %s

// CHECK-LABEL: @cmp_eq
// CHECK-SAME: (%[[LHS:.+]]: !vm.buffer, %[[RHS:.+]]: !vm.buffer)
util.func @cmp_eq(%lhs: !util.buffer, %rhs: !util.buffer) -> i1 {
  // CHECK: %[[RESULT:.+]] = vm.cmp.eq.ref %[[LHS]], %[[RHS]] : !vm.buffer
  %result = util.cmp.eq %lhs, %rhs : !util.buffer
  // CHECK: vm.return %[[RESULT]]
  util.return %result : i1
}

// -----

// CHECK-LABEL: @cmp_ne
// CHECK-SAME: (%[[LHS:.+]]: !vm.buffer, %[[RHS:.+]]: !vm.buffer)
util.func @cmp_ne(%lhs: !util.buffer, %rhs: !util.buffer) -> i1 {
  // CHECK: %[[RESULT:.+]] = vm.cmp.ne.ref %[[LHS]], %[[RHS]] : !vm.buffer
  %result = util.cmp.ne %lhs, %rhs : !util.buffer
  // CHECK: vm.return %[[RESULT]]
  util.return %result : i1
}
