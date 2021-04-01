// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK: vm.rodata @_exe_binary1_binary_ireebytecode dense<[0, 1, 2, 3]> : vector<4xi8>
// CHECK: vm.rodata @_exe_binary2_binary_spirv dense<[4, 5, 6, 7]> : vector<4xi8>
hal.executable @exe {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.binary @binary1 attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = 1230128453 : i32
  }
  hal.executable.binary @binary2 attributes {
    data = dense<[4, 5, 6, 7]> : vector<4xi8>,
    format = 1397773893 : i32
  }
}

// CHECK-LABEL: @executableCreate
func @executableCreate(
    // CHECK-SAME: %[[DEV:.+]]: !vm.ref<!hal.device>
    %device : !hal.device,
    // CHECK-SAME: %[[LAYOUT0:.+]]: !vm.ref<!hal.executable_layout>,
    %layout0 : !hal.executable_layout,
    // CHECK-SAME: %[[LAYOUT1:.+]]: !vm.ref<!hal.executable_layout>
    %layout1 : !hal.executable_layout
  ) -> (!hal.executable, !hal.executable) {
  // CHECK: %[[BIN0:.+]] = vm.const.ref.rodata @_exe_binary1_binary_ireebytecode : !vm.ref<!iree.byte_buffer>
  // CHECK: %[[EX0:.+]] = vm.call.variadic @hal.executable.create(
  // CHECK-SAME: %[[DEV]], %c1230128453, %[[BIN0]], [%[[LAYOUT0]], %[[LAYOUT1]]]
  // CHECK-SAME: ) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
  %0 = hal.executable.create device(%device : !hal.device) target(@exe::@binary1) layouts([%layout0, %layout1]) : !hal.executable
  // CHECK: %[[BIN1:.+]] = vm.const.ref.rodata @_exe_binary2_binary_spirv : !vm.ref<!iree.byte_buffer>
  // CHECK: %[[EX1:.+]] = vm.call.variadic @hal.executable.create(
  // CHECK-SAME: %[[DEV]], %c1397773893, %[[BIN1]], [%[[LAYOUT1]], %[[LAYOUT0]]]
  // CHECK-SAME: ) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
  %1 = hal.executable.create device(%device : !hal.device) target(@exe::@binary2) layouts([%layout1, %layout0]) : !hal.executable
  // CHECK: vm.return %[[EX0]], %[[EX1]]
  return %0, %1 : !hal.executable, !hal.executable
}
