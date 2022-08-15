// RUN: iree-opt --split-input-file --iree-vm-deduplicate-rodata %s | FileCheck %s

// CHECK-LABEL: vm.module public @basic
vm.module @basic {
  // CHECK: vm.rodata private @const0 dense<0>
  vm.rodata private @const0 dense<0> : vector<1xi8>
  // CHECK-NEXT: vm.rodata private @const1a dense<1>
  vm.rodata private @const1a dense<1> : vector<1xi8>
  vm.rodata private @const1b dense<1> : vector<1xi8>
  // CHECK-NEXT: vm.rodata private @const2 dense<2>
  vm.rodata private @const2 dense<2> : vector<1xi8>
  // CHECK-NEXT: vm.func @fn
  vm.func @fn() {
    // CHECK-NEXT: = vm.const.ref.rodata @const0 : !vm.buffer
    %v0 = vm.const.ref.rodata @const0 : !vm.buffer
    // CHECK-NEXT: = vm.const.ref.rodata @const1a : !vm.buffer
    %v1a = vm.const.ref.rodata @const1a : !vm.buffer
    // CHECK-NEXT: = vm.const.ref.rodata @const1a : !vm.buffer
    %v1b = vm.const.ref.rodata @const1b : !vm.buffer
    // CHECK-NEXT: = vm.const.ref.rodata @const2 : !vm.buffer
    %v2 = vm.const.ref.rodata @const2 : !vm.buffer
    vm.return
  }
}

// -----

// CHECK-LABEL: vm.module public @unique_mime_types
vm.module @unique_mime_types {
  // CHECK: vm.rodata private @const1a {mime_type = "aaa"} dense<1>
  vm.rodata private @const1a {mime_type = "aaa"} dense<1> : vector<1xi8>
  // CHECK: vm.rodata private @const1b {mime_type = "bbb"} dense<1>
  vm.rodata private @const1b {mime_type = "bbb"} dense<1> : vector<1xi8>
  // CHECK-NEXT: vm.func @fn
  vm.func @fn() {
    // CHECK-NEXT: = vm.const.ref.rodata @const1a : !vm.buffer
    %v1a = vm.const.ref.rodata @const1a : !vm.buffer
    // CHECK-NEXT: = vm.const.ref.rodata @const1b : !vm.buffer
    %v1b = vm.const.ref.rodata @const1b : !vm.buffer
    vm.return
  }
}

// -----

// CHECK-LABEL: vm.module public @widen_alignment
vm.module @widen_alignment {
  // CHECK: vm.rodata private @const1a {alignment = 16 : i64} dense<1>
  vm.rodata private @const1a {alignment = 1 : i64} dense<1> : vector<1xi8>
  vm.rodata private @const1b {alignment = 16 : i64} dense<1> : vector<1xi8>
  // CHECK-NEXT: vm.func @fn
  vm.func @fn() {
    // CHECK-NEXT: = vm.const.ref.rodata @const1a : !vm.buffer
    %v1a = vm.const.ref.rodata @const1a : !vm.buffer
    // CHECK-NEXT: = vm.const.ref.rodata @const1a : !vm.buffer
    %v1b = vm.const.ref.rodata @const1b : !vm.buffer
    vm.return
  }
}
