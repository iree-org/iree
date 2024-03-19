// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// This tests the vm.func conversion. Using the calling convention of EmitC,
// some arguments are getting added. For more details see comments on the
// `ConvertVMToEmitCPass` class in ConvertVMToEmitC.cpp.
vm.module @my_module {
  // CHECK: emitc.func private @my_module_fn(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>
  // CHECK-SAME:               %arg1: !emitc.ptr<!emitc.opaque<"struct my_module_t">>,
  // CHECK-SAME:               %arg2: !emitc.ptr<!emitc.opaque<"struct my_module_state_t">>,
  // CHECK-SAME:               %arg3: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>,
  // CHECK-SAME:               %arg4: i32,
  // CHECK-SAME:               %arg5: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>,
  // CHECK-SAME:               %arg6: !emitc.ptr<i32>)
  // CHECK-SAME:               -> !emitc.opaque<"iree_status_t">
  vm.func @fn(%arg0 : !vm.ref<?>, %arg1 : i32) -> (!vm.ref<?>, i32) {
    vm.return %arg0, %arg1 : !vm.ref<?>, i32
  }
}
