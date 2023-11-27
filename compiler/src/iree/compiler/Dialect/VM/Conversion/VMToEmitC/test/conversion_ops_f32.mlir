// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: @my_module_bitcast
vm.module @my_module {
  vm.func @bitcast(%arg0 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_bitcast_i32f32"(%arg3) : (i32) -> f32
    %0 = vm.bitcast.i32.f32 %arg0 : i32 -> f32
    // CHECK-NEXT: %1 = emitc.call_opaque "vm_bitcast_f32i32"(%0) : (f32) -> i32
    %1 = vm.bitcast.f32.i32 %0 : f32 -> i32
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @my_module_cast
vm.module @my_module {
  vm.func @cast(%arg0 : i32) -> (i32, i32) {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cast_si32f32"(%arg3) : (i32) -> f32
    %0 = vm.cast.si32.f32 %arg0 : i32 -> f32
    // CHECK-NEXT: %1 = emitc.call_opaque "vm_cast_ui32f32"(%arg3) : (i32) -> f32
    %1 = vm.cast.ui32.f32 %arg0 : i32 -> f32
    // CHECK-NEXT: %2 = emitc.call_opaque "vm_cast_f32si32"(%0) : (f32) -> i32
    %2 = vm.cast.f32.si32 %0 : f32 -> i32
    // CHECK-NEXT: %3 = emitc.call_opaque "vm_cast_f32ui32"(%1) : (f32) -> i32
    %3 = vm.cast.f32.ui32 %1 : f32 -> i32
    vm.return %2, %3 : i32, i32
  }
}
