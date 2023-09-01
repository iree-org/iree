// RUN: iree-compile --split-input-file --compile-mode=vm \
// RUN: --output-format=vm-c --iree-vm-c-module-optimize=false %s | FileCheck %s

vm.module @main_module attributes { version = 100 : i32 } {
  vm.import public @required.method0() attributes { minimum_version = 4 : i32 }
  vm.import public @required.method1() attributes { minimum_version = 5 : i32 }
  vm.import public optional @required.method2() attributes { minimum_version = 6 : i32 }

  vm.import public optional @optional.method0() attributes { minimum_version = 10 : i32 }
  vm.import public optional @optional.method1() attributes { minimum_version = 11 : i32 }
}

// CHECK: main_module_dependencies_[]
// CHECK: {"optional", 8}, 0, IREE_VM_MODULE_DEPENDENCY_FLAG_OPTIONAL
// CHECK: {"required", 8}, 5, IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED

// CHECK: main_module_imports_[]
// CHECK: {IREE_VM_NATIVE_IMPORT_REQUIRED, {"required.method0", 16}},
// CHECK: {IREE_VM_NATIVE_IMPORT_REQUIRED, {"required.method1", 16}},
// CHECK: {IREE_VM_NATIVE_IMPORT_OPTIONAL, {"required.method2", 16}},
// CHECK: {IREE_VM_NATIVE_IMPORT_OPTIONAL, {"optional.method0", 16}},
// CHECK: {IREE_VM_NATIVE_IMPORT_OPTIONAL, {"optional.method1", 16}},

// CHECK: main_module_descriptor_
// CHECK: {"main_module", 11}
// CHECK: 100
