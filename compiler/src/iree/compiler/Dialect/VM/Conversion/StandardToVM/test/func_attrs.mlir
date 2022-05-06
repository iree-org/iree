// RUN: iree-opt --split-input-file --pass-pipeline="test-iree-convert-std-to-vm" %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_iree_reflection
module @t001_iree_reflection {

module {
  // CHECK: vm.func private @t001_iree_reflection
  // CHECK-SAME: iree.reflection = {f = "FOOBAR"}
  func.func @t001_iree_reflection(%arg0: i32) -> (i32) attributes {
    iree.reflection = {f = "FOOBAR"}
  } {
    return %arg0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_iree_module_export_default
module @t002_iree_module_export_default {

module {
  // CHECK: vm.func private @internal_function_name
  // CHECK: vm.export @internal_function_name
  func.func @internal_function_name(%arg0: i32) -> (i32) {
    return %arg0 : i32
  }
}

}
