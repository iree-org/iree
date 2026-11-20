// RUN: iree-compile --compile-mode=vm \
// RUN: --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// Each internal function has its own source map, indexed by its ordinal.

// CHECK: "debug_database":
// CHECK: "functions": [
// CHECK-NEXT: "local_name": "first"
// CHECK: "local_name": "second"
// CHECK: "local_name": "third"
vm.module @module {
  vm.export @first
  vm.func @first(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }
  vm.export @second
  vm.func @second(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }
  vm.export @third
  vm.func @third(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }
}
