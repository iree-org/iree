// RUN: iree-compile --split-input-file --compile-mode=vm \
// RUN: --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK-LABEL: "main_module"
// CHECK: "version": 100
vm.module @main_module attributes { version = 100 : i32 } {
  // CHECK: "dependencies":

  // CHECK: "name": "required"
  // CHECK: "minimum_version": 5

  // CHECK: "name": "optional"
  // CHECK: "flags": "OPTIONAL"

  // CHECK: "imported_functions":

  // CHECK: "full_name": "required.method0"
  vm.import private @required.method0() attributes { minimum_version = 4 : i32 }
  // CHECK: "full_name": "required.method1"
  vm.import private @required.method1() attributes { minimum_version = 5 : i32 }
  // CHECK: "full_name": "required.method2"
  // CHECK: "flags": "OPTIONAL"
  vm.import private optional @required.method2() attributes { minimum_version = 6 : i32 }

  // CHECK: "full_name": "optional.method0"
  // CHECK: "flags": "OPTIONAL"
  vm.import private optional @optional.method0() attributes { minimum_version = 10 : i32 }
  // CHECK: "full_name": "optional.method1"
  // CHECK: "flags": "OPTIONAL"
  vm.import private optional @optional.method1() attributes { minimum_version = 11 : i32 }

}
