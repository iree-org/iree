// RUN: iree-opt -split-input-file -pass-pipeline='iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_iree_reflection
module @t001_iree_reflection {

module {
  // CHECK: func @t001_iree_reflection
  // CHECK-SAME: iree.reflection = {f = "FOOBAR"}
  func @t001_iree_reflection(%arg0: i32) -> (i32) attributes {iree.reflection = {f = "FOOBAR"}}
  {
    return %arg0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_iree_module_export_default
module @t002_iree_module_export_default {

module {
  // CHECK: func @internal_function_name
  // CHECK-NOT: iree.module.export
  // CHECK: vm.export @internal_function_name
  func @internal_function_name(%arg0: i32) -> (i32) attributes { iree.module.export }
  {
    return %arg0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t003_iree_module_export_alias
module @t003_iree_module_export_alias {

module {
  // CHECK: func @internal_function_name
  // CHECK-NOT: iree.module.export
  // CHECK: vm.export @internal_function_name as("external_function_name")
  func @internal_function_name(%arg0: i32) -> (i32)
      attributes { iree.module.export = "external_function_name" }
  {
    return %arg0 : i32
  }
}

}
