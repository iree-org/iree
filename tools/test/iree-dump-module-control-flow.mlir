// RUN: iree-compile \
// RUN:   %s \
// RUN:   --compile-mode=vm \
// RUN:   -o=%t.vmfb && \
// RUN: iree-dump-module \
// RUN:   --output=disassembly \
// RUN:   %t.vmfb | \
// RUN: FileCheck --check-prefix=CHECK %s
// RUN: iree-dump-module \
// RUN:   --output=disassembly \
// RUN:   --bytecode_offsets \
// RUN:   %t.vmfb | \
// RUN: FileCheck --check-prefix=OFFSETS %s

vm.module @dump_control_flow {
  vm.export @branch
  vm.func @branch(%cond: i32) -> i32 {
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    vm.cond_br %cond, ^left, ^right
  ^left:
    vm.br ^exit(%c1 : i32)
  ^right:
    vm.br ^exit(%c2 : i32)
  ^exit(%value : i32):
    vm.return %value : i32
  }
}

// CHECK: vm.module @dump_control_flow version 0
// CHECK-NEXT: vm.type i32
// CHECK-EMPTY:
// CHECK-NEXT: vm.export @branch
// CHECK-NEXT: vm.func @branch(%i0: i32) -> (i32) {
// CHECK: ^bb0:
// CHECK:   vm.cond_br %i0, ^bb1({{.*}}), ^bb1({{.*}})
// CHECK: ^bb1:
// OFFSETS: [00000000]^bb0:
