// Make sure that vm list ops are *not* unconditionally speculatable.

// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(vm.module(loop-invariant-code-motion))" %s | \
// RUN:   FileCheck %s

// CHECK-LABEL: @no_speculate
vm.module @no_speculate {
  // CHECK-LABEL: vm.func @list_alloc
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.list.alloc
  vm.func @list_alloc(%arg0: i32, %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.list.alloc %arg0 : (i32) -> !vm.list<i32>
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @list_size
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.list.size
  vm.func @list_size(%arg0: !vm.list<i32>, %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.list.size %arg0 : (!vm.list<i32>) -> i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @list_get
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.list.get.i32
  vm.func @list_get(%arg0: !vm.list<i32>, %arg1: i32,
                    %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.list.get.i32 %arg0, %arg1 : (!vm.list<i32>, i32) -> i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @list_set
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.list.set.i32
  vm.func @list_set(%arg0: !vm.list<i32>, %arg1: i32, %arg2: i32,
                    %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      vm.list.set.i32 %arg0, %arg1, %arg2 : (!vm.list<i32>, i32, i32)
    }
    vm.return
  }

}
