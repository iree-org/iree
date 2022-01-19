// RUN: iree-translate -iree-vm-ir-to-c-module -iree-vm-c-module-optimize=false %s | FileCheck %s

vm.module @control_flow_module {
  vm.func @control_flow_test(%a: i32, %cond: i32) -> i32 {
    vm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    vm.br ^bb3(%a: i32)
  ^bb2:
    %b = vm.add.i32 %a, %a : i32
    vm.br ^bb3(%b: i32)
  ^bb3(%c: i32):
    vm.br ^bb4(%c, %a : i32, i32)
  ^bb4(%d : i32, %e : i32):
    %0 = vm.add.i32 %d, %e : i32
    vm.return %0 : i32
  }
}
// CHECK: static iree_status_t control_flow_module_control_flow_test(iree_vm_stack_t* v1, control_flow_module_t* v2, control_flow_module_state_t* v3, int32_t [[A:[^ ]*]], int32_t [[COND:[^ ]*]], int32_t* [[RESULT:[^ ]*]]) {
  // CHECK-NEXT: int32_t [[COND_NZ:[^ ]*]];
  // CHECK-NEXT: bool [[COND_BOOL:[^ ]*]];
  // CHECK-NEXT: int32_t [[B:[^ ]*]];
  // CHECK-NEXT: int32_t [[V0:[^ ]*]];
  // CHECK-NEXT: iree_status_t [[STATUS:[^ ]*]];
  // CHECK-NEXT: int32_t [[C:[^ ]*]];
  // CHECK-NEXT: int32_t [[D:[^ ]*]];
  // CHECK-NEXT: int32_t [[E:[^ ]*]];
  // CHECK-NEXT: [[COND_NZ]] = vm_cmp_nz_i32([[COND]]);
  // CHECK-NEXT: [[COND_BOOL]] = EMITC_CAST([[COND_NZ]], bool);
  // CHECK-NEXT: if ([[COND_BOOL]]) {
  // CHECK-NEXT: goto [[BB1:[^ ]*]];
  // CHECK-NEXT: } else {
  // CHECK-NEXT: goto [[BB2:[^ ]*]];
  // CHECK-NEXT: }
  // CHECK-NEXT: [[BB1]]:
  // CHECK-NEXT: [[C]] = [[A]];
  // CHECK-NEXT: goto [[BB3:[^ ]*]];
  // CHECK-NEXT: [[BB2]]:
  // CHECK-NEXT: [[B]] = vm_add_i32([[A]], [[A]]);
  // CHECK-NEXT: [[C]] = [[B]];
  // CHECK-NEXT: goto [[BB3]];
  // CHECK-NEXT: [[BB3]]:
  // CHECK-NEXT: [[D]] = [[C]];
  // CHECK-NEXT: [[E]] = [[A]];
  // CHECK-NEXT: goto [[BB4:[^ ]*]];
  // CHECK-NEXT: [[BB4]]:
  // CHECK-NEXT: [[V0]] = vm_add_i32([[D]], [[E]]);
  // CHECK-NEXT: EMITC_DEREF_ASSIGN_VALUE([[RESULT]], [[V0]]);
  // CHECK-NEXT: [[STATUS]] = iree_ok_status();
  // CHECK-NEXT: return [[STATUS]];
