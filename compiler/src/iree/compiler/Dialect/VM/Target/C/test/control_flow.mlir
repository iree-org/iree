// RUN: iree-compile --compile-mode=vm --output-format=vm-c --iree-vm-c-module-optimize=false %s | FileCheck %s

vm.module @control_flow_module {
  vm.func @control_flow_test(%a: i32, %cond: i32) -> i32 {
    vm.cond_br %cond, ^bb2(%a, %a : i32, i32), ^bb1
  ^bb1:
    %b = vm.add.i32 %a, %a : i32
    vm.br ^bb2(%b, %a : i32, i32)
  ^bb2(%c: i32, %d: i32):
    %e = vm.add.i32 %c, %d : i32
    vm.return %e : i32
  }
}
// CHECK: iree_status_t control_flow_module_control_flow_test(iree_vm_stack_t* v1, struct control_flow_module_t* v2, struct control_flow_module_state_t* v3, int32_t [[A:[^ ]*]], int32_t [[COND:[^ ]*]], int32_t* [[RESULT:[^ ]*]]) {
  // CHECK-NEXT: int32_t [[COND_NZ:[^ ]*]];
  // CHECK-NEXT: bool [[COND_BOOL:[^ ]*]];
  // CHECK-NEXT: int32_t [[B:[^ ]*]];
  // CHECK-NEXT: int32_t [[V0:[^ ]*]];
  // CHECK-NEXT: iree_status_t [[STATUS:[^ ]*]];
  // CHECK-NEXT: int32_t [[C:[^ ]*]];
  // CHECK-NEXT: int32_t [[D:[^ ]*]];
  // CHECK-NEXT: [[COND_NZ]] = vm_cmp_nz_i32([[COND]]);
  // CHECK-NEXT: [[COND_BOOL]] = (bool) [[COND_NZ]];
  // CHECK-NEXT: if ([[COND_BOOL]]) {
  // CHECK-NEXT: [[C]] = [[A]];
  // CHECK-NEXT: [[D]] = [[A]];
  // CHECK-NEXT: goto [[BB2:[^ ]*]];
  // CHECK-NEXT: } else {
  // CHECK-NEXT: goto [[BB1:[^ ]*]];
  // CHECK-NEXT: }
  // CHECK-NEXT: [[BB1]]:
  // CHECK-NEXT: [[B]] = vm_add_i32([[A]], [[A]]);
  // CHECK-NEXT: [[C]] = [[B]];
  // CHECK-NEXT: [[D]] = [[A]];
  // CHECK-NEXT: goto [[BB2:[^ ]*]];
  // CHECK-NEXT: [[BB2]]:
  // CHECK-NEXT: [[V0]] = vm_add_i32([[C]], [[D]]);
  // CHECK-NEXT: EMITC_DEREF_ASSIGN_VALUE([[RESULT]], [[V0]]);
  // CHECK-NEXT: [[STATUS]] = iree_ok_status();
  // CHECK-NEXT: return [[STATUS]];
