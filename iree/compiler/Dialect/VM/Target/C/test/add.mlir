// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: // c module stub
// CHECK: %0 = emitc.call "vm_add_i32"(%arg0, %arg1) {args = [0 : index, 1 : index]} : (i32, i32) -> i32
vm.module @add_module {
  vm.func @add_i32(%arg0 : i32, %arg1 : i32) {
    %0 = vm.add.i32 %arg0, %arg1 : i32
    vm.return
  }
}
