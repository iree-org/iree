// RUN: iree-opt --split-input-file --allow-unregistered-dialect --pass-pipeline="builtin.module(test-iree-convert-std-to-vm)" --iree-vm-target-index-bits=32 %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_iree_reflection
module @t001_iree_reflection {
module {
  // CHECK: vm.func private @t001_iree_reflection
  // CHECK-SAME: iree.reflection = {f = "FOOBAR"}
  func.func @t001_iree_reflection(%arg0: i32) -> i32 attributes {
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
  func.func @internal_function_name(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}
}

// -----
// CHECK-LABEL: @t003_extern_func
module @t003_extern_func {
module {
  // CHECK: vm.import private @some.import(i32) -> !vm.buffer
  // CHECK-SAME: attributes {minimum_version = 4 : i32}
  func.func private @some.import(%arg0: index) -> !util.buffer attributes {
    vm.version = 4 : i32
  }
  // CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> !vm.buffer
  func.func @my_fn(%arg0: index) -> !util.buffer {
    // CHECK: %[[RET:.+]] = vm.call @some.import(%[[ARG0]]) : (i32) -> !vm.buffer
    %0 = call @some.import(%arg0) : (index) -> !util.buffer
    // CHECK: return %[[RET]]
    return %0 : !util.buffer
  }
}
}

// -----
// CHECK-LABEL: @t003_extern_func_after
module @t003_extern_func_after {
module {
  // CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> !vm.buffer
  func.func @my_fn(%arg0: index) -> !util.buffer {
    // CHECK: %[[RET:.+]] = vm.call @some.import(%[[ARG0]]) : (i32) -> !vm.buffer
    %0 = call @some.import(%arg0) : (index) -> !util.buffer
    // CHECK: return %[[RET]]
    return %0 : !util.buffer
  }
  // CHECK: vm.import private @some.import(i32) -> !vm.buffer
  func.func private @some.import(%arg0: index) -> !util.buffer
}
}

// -----
// CHECK-LABEL: @t004_extern_func_signature
module @t004_extern_func_signature {
module {
  // CHECK: vm.import private @some.import(i64) -> i64
  func.func private @some.import(%arg0: index) -> index attributes {
    vm.signature = (i64) -> i64
  }
  // CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> i32
  func.func @my_fn(%arg0: index) -> index {
    // CHECK: %[[IMPORT_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
    // CHECK: %[[IMPORT_RET:.+]] = vm.call @some.import(%[[IMPORT_ARG]]) : (i64) -> i64
    %0 = call @some.import(%arg0) : (index) -> index
    // CHECK: %[[RET:.+]] = vm.trunc.i64.i32 %[[IMPORT_RET]]
    // CHECK: return %[[RET]] : i32
    return %0 : index
  }
}
}

// -----
// CHECK-LABEL: @t004_extern_func_signature_after
module @t004_extern_func_signature_after {
module {
  // CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> i32
  func.func @my_fn(%arg0: index) -> index {
    // CHECK: %[[IMPORT_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
    // CHECK: %[[IMPORT_RET:.+]] = vm.call @some.import(%[[IMPORT_ARG]]) : (i64) -> i64
    %0 = call @some.import(%arg0) : (index) -> index
    // CHECK: %[[RET:.+]] = vm.trunc.i64.i32 %[[IMPORT_RET]]
    // CHECK: return %[[RET]] : i32
    return %0 : index
  }
  // CHECK: vm.import private @some.import(i64) -> i64
  func.func private @some.import(%arg0: index) -> index attributes {
    vm.signature = (i64) -> i64
  }
}
}

// -----
// CHECK-LABEL: @t005_call
module @t005_call {
module {
  func.func private @other.fn(%arg0: i32) -> i32
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0 : i32) -> i32 {
    // CHECK: vm.call @other.fn(%[[ARG0]]) : (i32) -> i32
    %0 = call @other.fn(%arg0) : (i32) -> i32
    return %0 : i32
  }
}
}

// -----
// CHECK-LABEL: @t005_call_int_promotion
module @t005_call_int_promotion {
module {
  func.func private @other.fn(%arg0: i1) -> i1
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0 : i1) -> i1 {
    // CHECK: vm.call @other.fn(%[[ARG0]]) : (i32) -> i32
    %0 = call @other.fn(%arg0) : (i1) -> i1
    return %0 : i1
  }
}
}

// -----
// CHECK-LABEL: @t006_external_call_fallback
module @t006_external_call_fallback {
module {
  // NOTE: we require conversion for the import but not the fallback!
  // CHECK: vm.import private optional @some.import(i64) -> i64
  func.func private @some.import(%arg0: index) -> index attributes {
    vm.signature = (i64) -> i64,
    vm.fallback = @some_fallback
  }
  // CHECK: vm.func private @some_fallback(%{{.+}}: i32) -> i32
  func.func private @some_fallback(%arg0: index) -> index {
    return %arg0 : index
  }
  // CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> i32
  func.func @my_fn(%arg0: index) -> index {
    // CHECK: %[[RESOLVED:.+]] = vm.import.resolved @some.import
    // CHECK: vm.cond_br %[[RESOLVED]], ^bb1, ^bb2
    %0 = call @some.import(%arg0) : (index) -> index

    // CHECK: ^bb1:
    // CHECK: %[[IMPORT_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
    // CHECK: %[[IMPORT_RET:.+]] = vm.call @some.import(%[[IMPORT_ARG]]) : (i64) -> i64
    // CHECK: %[[BB1_RET:.+]] = vm.trunc.i64.i32 %[[IMPORT_RET]]
    // CHECK: vm.br ^bb3(%[[BB1_RET]] : i32)

    // CHECK: ^bb2:
    // CHECK: %[[FALLBACK_RET:.+]] = vm.call @some_fallback(%[[ARG0]]) : (i32) -> i32
    // CHECK: vm.br ^bb3(%[[FALLBACK_RET]] : i32)

    // CHECK: ^bb3(%[[RET:.+]]: i32):
    // CHECK: return %[[RET]]
    return %0 : index
  }
}
}

// -----
// CHECK-LABEL: @t007_external_call_fallback_import
module @t007_external_call_fallback_import {
module {
  // NOTE: we require conversion for the import but not the fallback!
  // CHECK: vm.import private optional @some.import(i64) -> i64
  func.func private @some.import(%arg0: index) -> index attributes {
    vm.signature = (i64) -> i64,
    vm.fallback = @other.fallback
  }
  // CHECK: vm.import private @other.fallback(i64) -> i64
  func.func private @other.fallback(%arg0: index) -> index attributes {
    vm.signature = (i64) -> i64
  }
  // CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> i32
  func.func @my_fn(%arg0: index) -> index {
    // CHECK: %[[RESOLVED:.+]] = vm.import.resolved @some.import
    // CHECK: vm.cond_br %[[RESOLVED]], ^bb1, ^bb2
    %0 = call @some.import(%arg0) : (index) -> index

    // CHECK: ^bb1:
    // CHECK: %[[IMPORT_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
    // CHECK: %[[IMPORT_RET:.+]] = vm.call @some.import(%[[IMPORT_ARG]]) : (i64) -> i64
    // CHECK: %[[BB1_RET:.+]] = vm.trunc.i64.i32 %[[IMPORT_RET]]
    // CHECK: vm.br ^bb3(%[[BB1_RET]] : i32)

    // CHECK: ^bb2:
    // CHECK: %[[FALLBACK_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
    // CHECK: %[[FALLBACK_RET:.+]] = vm.call @other.fallback(%[[FALLBACK_ARG]]) : (i64) -> i64
    // CHECK: %[[BB2_RET:.+]] = vm.trunc.i64.i32 %[[FALLBACK_RET]]
    // CHECK: vm.br ^bb3(%[[BB2_RET]] : i32)

    // CHECK: ^bb3(%[[RET:.+]]: i32):
    // CHECK: return %[[RET]]
    return %0 : index
  }
}
}

// -----
// CHECK-LABEL: @t007_extern_func_opaque_types
module @t007_extern_func_opaque_types {
module {
  // CHECK: vm.import private @some.import() -> !vm.ref<!some.type<foo>>
  func.func private @some.import() -> !some.type<foo>
  // CHECK: vm.func private @my_fn() -> !vm.ref<!some.type<foo>>
  func.func @my_fn() -> !some.type<foo> {
    // CHECK: %[[RET:.+]] = vm.call @some.import() : () -> !vm.ref<!some.type<foo>>
    %0 = call @some.import() : () -> !some.type<foo>
    // CHECK: return %[[RET]]
    return %0 : !some.type<foo>
  }
}
}
