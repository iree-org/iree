// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc)' %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @my_module_branch_empty
  vm.func @branch_empty() {
    // CHECK: cf.br ^bb1
    vm.br ^bb1
  ^bb1:
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_branch_int_args
  vm.func @branch_int_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: cf.br ^bb1(%arg3, %arg4 : i32, i32)
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_branch_ref_args
  vm.func @branch_ref_args(%arg0 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: cf.br ^bb1
    // CHECK: cf.br ^bb2
    vm.br ^bb1(%arg0 : !vm.ref<?>)
  ^bb1(%0 : !vm.ref<?>):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : !vm.ref<?>
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_branch_mixed_args
  vm.func @branch_mixed_args(%arg0 : !vm.ref<?>, %arg1: i32, %arg2 : !vm.ref<?>, %arg3: i32) -> !vm.ref<?> {
    // CHECK: cf.br ^bb1
    // CHECK: cf.br ^bb2(%arg4, %arg6 : i32, i32)
    vm.br ^bb1(%arg0, %arg1, %arg2, %arg3 : !vm.ref<?>, i32, !vm.ref<?>, i32)
  ^bb1(%0 : !vm.ref<?>, %1 : i32, %2 : !vm.ref<?>, %3 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : !vm.ref<?>
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_cond_branch_empty
  vm.func @cond_branch_empty(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: cf.cond_br %{{.}}, ^bb1, ^bb2
    vm.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    // CHECK-NOT: vm.return
    // CHECK: return
    vm.return %arg1 : i32
  ^bb2:
    // CHECK-NOT: vm.return
    // CHECK: return
    vm.return %arg2 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_cond_branch_int_args
  vm.func @cond_branch_int_args(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: cf.cond_br {{%.}}, ^bb1(%arg4 : i32), ^bb2(%arg5 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : i32
  ^bb2(%1 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %1 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_cond_branch_ref_args
  vm.func @cond_branch_ref_args(%arg0 : i32, %arg1 : !vm.ref<?>, %arg2 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: cf.cond_br {{%.}}, ^bb1, ^bb4
    // CHECK: cf.br ^bb2
    // CHEKC: cf.br ^bb3
    vm.cond_br %arg0, ^bb1(%arg1 : !vm.ref<?>), ^bb2(%arg2 : !vm.ref<?>)
  ^bb1(%0 : !vm.ref<?>):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : !vm.ref<?>
  ^bb2(%1 : !vm.ref<?>):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %1 : !vm.ref<?>
  }
}
