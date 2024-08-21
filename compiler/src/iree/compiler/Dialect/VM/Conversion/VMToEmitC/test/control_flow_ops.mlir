// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_branch_empty
  vm.func @branch_empty() {
    // CHECK: cf.br ^bb1
    vm.br ^bb1
  ^bb1:
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_branch_int_args
  vm.func @branch_int_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: cf.br ^bb1(%arg3, %arg4 : i32, i32)
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_branch_ref_args
  vm.func @branch_ref_args(%arg0 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: cf.br ^bb1
    // CHECK: cf.br ^bb2
    vm.br ^bb1(%arg0 : !vm.ref<?>)
  ^bb1(%0 : !vm.ref<?>):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : !vm.ref<?>
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_branch_mixed_args
  vm.func @branch_mixed_args(%arg0 : !vm.ref<?>, %arg1: i32, %arg2 : !vm.ref<?>, %arg3: i32) -> !vm.ref<?> {
    // CHECK: cf.br ^bb1
    // CHECK: cf.br ^bb2(%arg4, %arg6 : i32, i32)
    vm.br ^bb1(%arg0, %arg1, %arg2, %arg3 : !vm.ref<?>, i32, !vm.ref<?>, i32)
  ^bb1(%0 : !vm.ref<?>, %1 : i32, %2 : !vm.ref<?>, %3 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : !vm.ref<?>
  }
}

// -----

// Test vm.call conversion on an imported function.
vm.module @my_module {
  // CHECK: emitc.func private @my_module_call_[[IMPORTFN:[^\(]+]]
  vm.import private @imported_fn(%arg0 : i32) -> i32

  // CHECK: emitc.func private @my_module_call_imported_fn
  vm.func @call_imported_fn(%arg0 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS_VAR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORTS:.+]] = "emitc.member_of_ptr"(%arg2) <{member = "imports"}> : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: emitc.assign %[[IMPORTS]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">> to %[[IMPORTS_VAR]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS_VAR]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // Create a variable for the function result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = emitc.call @my_module_call_[[IMPORTFN]](%arg0, %[[IMPORT]], %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>, i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @imported_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test that the order of imports and calls doesn't matter.
vm.module @my_module {
  // CHECK: emitc.func private @my_module_call_imported_fn
  vm.func @call_imported_fn(%arg0 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS_VAR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORTS:.+]] = "emitc.member_of_ptr"(%arg2) <{member = "imports"}> : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: emitc.assign %[[IMPORTS]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">> to %[[IMPORTS_VAR]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS_VAR]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // Create a variable for the function result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = emitc.call @my_module_call_[[IMPORTFN:[^\(]+]](%arg0, %[[IMPORT]], %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>, i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @imported_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }

  // CHECK: emitc.func private @my_module_call_[[IMPORTFN]]
  vm.import private @imported_fn(%arg0 : i32) -> i32
}

// -----

// Test vm.call conversion on an internal function.
vm.module @my_module {
  vm.func @internal_fn(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }

  // CHECK-LABEL: emitc.func private @my_module_call_internal_fn
  vm.func @call_internal_fn(%arg0 : i32) -> i32 {

    // Create a variable for the result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = emitc.call @my_module_internal_fn(%arg0, %arg1, %arg2, %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"struct my_module_t">>,
    // CHECK-SAME:        !emitc.ptr<!emitc.opaque<"struct my_module_state_t">>, i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @internal_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test vm.call.variadic conversion on an imported function.
vm.module @my_module {
  // CHECK: emitc.func private @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import private @variadic_fn(%arg0 : i32 ...) -> i32

  // CHECK: emitc.func private @my_module_call_variadic
  vm.func @call_variadic(%arg0 : i32, %arg1 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS_VAR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORTS:.+]] = "emitc.member_of_ptr"(%arg2) <{member = "imports"}> : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: emitc.assign %[[IMPORTS]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">> to %[[IMPORTS_VAR]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS_VAR]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // This holds the number of variadic arguments.
    // CHECK-NEXT: %[[NARGS:.+]] = "emitc.constant"() <{value = 2 : i32}> : () -> i32

    // Create a variable for the result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = emitc.call @my_module_call_[[VARIADICFN]](%arg0, %[[IMPORT]], %[[NARGS]], %arg3, %arg4, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
    // CHECK-SAME:        i32, i32, i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call.variadic @variadic_fn([%arg0, %arg1]) : (i32 ...) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test vm.call.variadic with zero arguments.
vm.module @my_module {
  // CHECK: emitc.func private @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import private @variadic_fn(%arg0 : i32 ...) -> i32

  // CHECK: emitc.func private @my_module_call_variadic
  vm.func @call_variadic() -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS_VAR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORTS:.+]] = "emitc.member_of_ptr"(%arg2) <{member = "imports"}> : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: emitc.assign %[[IMPORTS]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">> to %[[IMPORTS_VAR]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS_VAR]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // This holds the number of variadic arguments.
    // CHECK-NEXT: %[[NARGS:.+]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32

    // Create a variable for the result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = emitc.call @my_module_call_[[VARIADICFN]](%arg0, %[[IMPORT]], %[[NARGS]], %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
    // CHECK-SAME:        i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call.variadic @variadic_fn([]) : (i32 ...) -> i32
    vm.return %0 : i32
  }
}

// -----

// TODO(simon-camp): add check statements
// Test vm.call.variadic with multiple variadic packs.
vm.module @my_module {
  // CHECK: emitc.func private @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import private @variadic_fn(%is : i32 ..., %fs : f32 ...) -> i32

  // CHECK: emitc.func private @my_module_call_variadic
  vm.func @call_variadic(%i : i32, %f : f32) -> i32 {

    %0 = vm.call.variadic @variadic_fn([%i, %i], [%f, %f, %f]) : (i32 ..., f32 ...) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_cond_branch_empty
  vm.func @cond_branch_empty(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: cf.cond_br %{{.}}, ^bb1, ^bb2
    vm.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    // CHECK-NOT: vm.return
    // CHECK: return
    vm.return %arg1 : i32
  ^bb2:
    // CHECK-NOT: vm.return
    // CHECK: return
    vm.return %arg2 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_cond_branch_int_args
  vm.func @cond_branch_int_args(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: cf.cond_br {{%.}}, ^bb1(%arg4 : i32), ^bb2(%arg5 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : i32
  ^bb2(%1 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %1 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_cond_branch_ref_args
  vm.func @cond_branch_ref_args(%arg0 : i32, %arg1 : !vm.ref<?>, %arg2 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: cf.cond_br {{%.}}, ^bb1, ^bb4
    // CHECK: cf.br ^bb2
    // CHECK: cf.br ^bb3
    vm.cond_br %arg0, ^bb1(%arg1 : !vm.ref<?>), ^bb2(%arg2 : !vm.ref<?>)
  ^bb1(%0 : !vm.ref<?>):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : !vm.ref<?>
  ^bb2(%1 : !vm.ref<?>):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %1 : !vm.ref<?>
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_br_table_empty
vm.module @my_module {
  vm.func @br_table_empty(%arg0: i32, %arg1: i32) -> i32 {
    //  CHECK-NOT: vm.br_table
    //      CHECK:  cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:  cf.br ^bb2(%arg4 : i32)
    // CHECK-NEXT: ^bb2(%0: i32):
    //      CHECK:  return
    vm.br_table %arg0 {
      default: ^bb1(%arg1 : i32)
    }
  ^bb1(%0 : i32):
    // CHECK: return
    // CHECK-NOT: vm.return
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_br_table
vm.module @my_module {
  vm.func @br_table(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    //  CHECK-NOT: vm.br_table
    //      CHECK:  cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    //      CHECK:  emitc.call_opaque "vm_cmp_eq_i32"
    //      CHECK:  emitc.call_opaque "vm_cmp_nz_i32"
    //      CHECK:  cf.cond_br %{{.+}}, ^bb5(%arg4 : i32), ^bb2
    //      CHECK: ^bb2:
    //      CHECK:  emitc.call_opaque "vm_cmp_eq_i32"
    //      CHECK:  emitc.call_opaque "vm_cmp_nz_i32"
    //      CHECK:  cf.cond_br %{{.+}}, ^bb5(%arg5 : i32), ^bb3
    //      CHECK: ^bb3:
    //      CHECK:  cf.br ^bb4(%arg3 : i32)
    vm.br_table %arg0 {
      default: ^bb1(%arg0 : i32),
      0: ^bb2(%arg1 : i32),
      1: ^bb2(%arg2 : i32)
    }
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_fail
  vm.func @fail(%arg0 : i32) {

    // Typecast the argument to fail and branch respectively.
    // CHECK-NEXT: %[[COND:.+]] = emitc.cast %arg3 : i32 to i1
    // CHECK-NEXT: cf.cond_br %[[COND]], ^[[FAIL:.+]], ^[[OK:.+]]

    // In case of success, return ok status.
    // CHECK-NEXT: ^[[OK]]:
    // CHECK-NEXT: %[[OKSTATUS:.+]] = emitc.call_opaque "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %[[OKSTATUS]] : !emitc.opaque<"iree_status_t">

    // In case of fail, return status message.
    // CHECK-NEXT: ^[[FAIL]]:
    // CHECK-NEXT: %[[MSG:.+]] = emitc.call_opaque "iree_make_cstring_view"() {args = [#emitc.opaque<"\22message\22">]}
    // CHECK-SAME:     : () -> !emitc.opaque<"iree_string_view_t">
    // CHECK-NEXT: %[[MSGSIZE_VAR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: %[[MSGSIZE:.+]] = "emitc.member"(%[[MSG]]) <{member = "size"}> : (!emitc.opaque<"iree_string_view_t">) -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: emitc.assign %[[MSGSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[MSGSIZE_VAR]] : !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: %[[MSGSIZEINT:.+]] = emitc.cast %[[MSGSIZE_VAR]] : !emitc.opaque<"iree_host_size_t"> to !emitc.opaque<"int">
    // CHECK-NEXT: %[[MSGDATA_VAR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"const char">>
    // CHECK-NEXT: %[[MSGDATA:.+]] = "emitc.member"(%[[MSG]]) <{member = "data"}> : (!emitc.opaque<"iree_string_view_t">) -> !emitc.ptr<!emitc.opaque<"const char">>
    // CHECK-NEXT: emitc.assign %[[MSGDATA]] : !emitc.ptr<!emitc.opaque<"const char">> to %[[MSGDATA_VAR]] : !emitc.ptr<!emitc.opaque<"const char">>
    // CHECK-NEXT: %[[FAILSTATUS:.+]] = emitc.call_opaque "iree_status_allocate_f"(%[[MSGSIZEINT]], %[[MSGDATA_VAR]])
    // CHECK-SAME:     {args = [#emitc.opaque<"IREE_STATUS_FAILED_PRECONDITION">, #emitc.opaque<"\22<vm>\22">, 0 : i32, #emitc.opaque<"\22%.*s\22">, 0 : index, 1 : index]}
    // CHECK-SAME:     : (!emitc.opaque<"int">, !emitc.ptr<!emitc.opaque<"const char">>) -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %[[FAILSTATUS]] : !emitc.opaque<"iree_status_t">
    vm.fail %arg0, "message"
  }
}

// -----

// Test vm.import conversion on a void function.
vm.module @my_module {

  // CHECK-LABEL: emitc.func private @my_module_call_0v_v_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {specifiers = ["static"]} {

  // Calculate the size of the arguments. To avoid empty structs we insert a dummy value.
  // CHECK-NEXT: %[[ARGSIZE:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">

  // Calculate the size of the result. To avoid empty structs we insert a dummy value.
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "function"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.assign %[[ARGSTRUCTFN]] : !emitc.opaque<"iree_vm_function_t"> to %[[ARGSTRUCTFN_MEMBER]] : !emitc.opaque<"iree_vm_function_t">

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "arguments"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.apply "&"(%[[ARGBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[ARGSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[ARGSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSDATA:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGBYTESPANDATA]] : !emitc.ptr<ui8> to %[[ARGSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.apply "&"(%[[RESBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[RESULTSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[RESSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESSDATA:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESBYTESPANDATA]] : !emitc.ptr<ui8> to %[[RESSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Check that we don't pack anything into the argument struct.
  // CHECK-NOT: "emitc.member"(%[[RESBYTESPAN]]) <{member = "arguments"}>
  // CHECK-NOT: "emitc.member"(%{{.+}}) <{member = "data"}>

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[MODULE_MEMBER:.+]] = "emitc.member_of_ptr"(%arg1) <{member = "module"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: emitc.assign %[[MODULE_MEMBER]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">> to %[[IMPORTMOD]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[BEGIN_CALL:.+]] = "emitc.member_of_ptr"(%[[IMPORTMOD]]) <{member = "begin_call"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_module_t">>) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_CALL_INDIRECT"(%[[BEGIN_CALL]], %[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])

  // Check that we don't unpack anything from the result struct.
  // CHECK-NOT: "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}>
  // CHECK-NOT: "emitc.member"(%{{.+}}) <{member = "data"}>

  // Return ok status.
  //      CHECK: %[[OK:.+]] = emitc.call_opaque "iree_ok_status"()
  // CHECK-NEXT: return %[[OK]]
  vm.import private @ref_fn() -> ()

  vm.func @import_ref() -> () {
    vm.call @ref_fn() : () -> ()
    vm.return
  }
}

// -----

// Test vm.import conversion on a variadic function.
vm.module @my_module {

  // CHECK-LABEL: emitc.func private @my_module_call_0iCiD_i_2_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
  // CHECK-SAME:                                             %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: !emitc.ptr<i32>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {specifiers = ["static"]} {

  // Calculate the size of the arguments.
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE01:.+]] = emitc.add %[[ARGSIZE0]], %[[ARGSIZE1]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE012:.+]] = emitc.add %[[ARGSIZE01]], %[[ARGSIZE2]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE3:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE0123:.+]] = emitc.add %[[ARGSIZE012]], %[[ARGSIZE3]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE4:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.add %[[ARGSIZE0123]], %[[ARGSIZE4]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.add %[[RESULTSIZE0]], %[[RESULTSIZE1]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "function"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.assign %[[ARGSTRUCTFN]] : !emitc.opaque<"iree_vm_function_t"> to %[[ARGSTRUCTFN_MEMBER]] : !emitc.opaque<"iree_vm_function_t">

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "arguments"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.apply "&"(%[[ARGBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[ARGSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[ARGSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSDATA:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGBYTESPANDATA]] : !emitc.ptr<ui8> to %[[ARGSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.apply "&"(%[[RESBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[RESULTSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[RESSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESSDATA:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESBYTESPANDATA]] : !emitc.ptr<ui8> to %[[RESSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Pack the arguments into the struct.
  // Here we also create pointers for non-pointer types.
  // CHECK-NEXT: %[[ARGS:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGS_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "arguments"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: emitc.assign %[[ARGS_MEMBER]] : !emitc.opaque<"iree_byte_span_t"> to %[[ARGS]] : !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGSPTR_MEMBER:.+]] = "emitc.member"(%[[ARGS]]) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGSPTR_MEMBER]] : !emitc.ptr<ui8> to %[[ARGSPTR]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGHOSTSIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1PTR:.+]] = emitc.apply "&"(%arg2) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[ARGSPTR]], %[[A1PTR]], %[[ARGHOSTSIZE]])
  // CHECK-NEXT: %[[ARGHOSTSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1ADDR:.+]] = emitc.add %[[ARGSPTR]], %[[ARGHOSTSIZE2]]
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A1SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2PTR:.+]] = emitc.apply "&"(%arg3) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A1ADDR]], %[[A2PTR]], %[[A1SIZE]])
  // CHECK-NEXT: %[[A1SIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2ADDR:.+]] = emitc.add %[[A1ADDR]], %[[A1SIZE2]]
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A2SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A3PTR:.+]] = emitc.apply "&"(%arg4) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A2ADDR]], %[[A3PTR]], %[[A2SIZE]])
  // CHECK-NEXT: %[[A2SIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A3ADDR:.+]] = emitc.add %[[A2ADDR]], %[[A2SIZE2]]
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A3SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A4PTR:.+]] = emitc.apply "&"(%arg5) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A3ADDR]], %[[A4PTR]], %[[A3SIZE:.+]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[MODULE_MEMBER:.+]] = "emitc.member_of_ptr"(%arg1) <{member = "module"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: emitc.assign %[[MODULE_MEMBER]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">> to %[[IMPORTMOD]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[BEGIN_CALL:.+]] = "emitc.member_of_ptr"(%[[IMPORTMOD]]) <{member = "begin_call"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_module_t">>) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_CALL_INDIRECT"(%[[BEGIN_CALL]], %[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RES_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: emitc.assign %[[RES_MEMBER]] : !emitc.opaque<"iree_byte_span_t"> to %[[RES]] : !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESPTR_MEMBER:.+]] = "emitc.member"(%[[RES]]) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESPTR_MEMBER]] : !emitc.ptr<ui8> to %[[RESPTR]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESHOSTSIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%arg6, %[[RESPTR]], %[[RESHOSTSIZE]])

  // Return ok status.
  // CHECK-NEXT: %[[OK:.+]] = emitc.call_opaque "iree_ok_status"()
  // CHECK-NEXT: return %[[OK]]
  vm.import private @variadic_fn(%arg0 : i32, %arg1 : i32 ...) -> i32

  vm.func @import_variadic(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    %0 = vm.call.variadic @variadic_fn(%arg0, [%arg1, %arg2]) : (i32, i32 ...) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test vm.call.variadic with zero variadic arguments.
vm.module @my_module {

  // CHECK-LABEL: emitc.func private @my_module_call_0iCiD_i_0_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
  // CHECK-SAME:                                             %arg2: i32, %arg3: i32, %arg4: !emitc.ptr<i32>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {specifiers = ["static"]} {

  // Calculate the size of the arguments.
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE01:.+]] = emitc.add %[[ARGSIZE0]], %[[ARGSIZE1]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.add %[[ARGSIZE01]], %[[ARGSIZE2]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.add %[[RESULTSIZE0]], %[[RESULTSIZE1]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "function"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.assign %[[ARGSTRUCTFN]] : !emitc.opaque<"iree_vm_function_t"> to %[[ARGSTRUCTFN_MEMBER]] : !emitc.opaque<"iree_vm_function_t">

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "arguments"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.apply "&"(%[[ARGBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[ARGSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[ARGSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSDATA:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGBYTESPANDATA]] : !emitc.ptr<ui8> to %[[ARGSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.apply "&"(%[[RESBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[RESULTSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[RESSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESSDATA:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESBYTESPANDATA]] : !emitc.ptr<ui8> to %[[RESSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Pack the arguments into the struct.
  // Here we also create pointers for non-pointer types.
  // CHECK-NEXT: %[[ARGS:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGS_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "arguments"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: emitc.assign %[[ARGS_MEMBER]] : !emitc.opaque<"iree_byte_span_t"> to %[[ARGS]] : !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGSPTR_MEMBER:.+]] = "emitc.member"(%[[ARGS]]) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGSPTR_MEMBER]] : !emitc.ptr<ui8> to %[[ARGSPTR]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGHOSTSIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1PTR:.+]] = emitc.apply "&"(%arg2) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[ARGSPTR]], %[[A1PTR]], %[[ARGHOSTSIZE]])
  // CHECK-NEXT: %[[ARGHOSTSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1ADDR:.+]] = emitc.add %[[ARGSPTR]], %[[ARGHOSTSIZE2]]
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A1SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2PTR:.+]] = emitc.apply "&"(%arg3) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A1ADDR]], %[[A2PTR]], %[[A1SIZE]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[MODULE_MEMBER:.+]] = "emitc.member_of_ptr"(%arg1) <{member = "module"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: emitc.assign %[[MODULE_MEMBER]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">> to %[[IMPORTMOD]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[BEGIN_CALL:.+]] = "emitc.member_of_ptr"(%[[IMPORTMOD]]) <{member = "begin_call"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_module_t">>) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_CALL_INDIRECT"(%[[BEGIN_CALL]], %[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RES_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: emitc.assign %[[RES_MEMBER]] : !emitc.opaque<"iree_byte_span_t"> to %[[RES]] : !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESPTR_MEMBER:.+]] = "emitc.member"(%[[RES]]) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESPTR_MEMBER]] : !emitc.ptr<ui8> to %[[RESPTR]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESHOSTSIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%arg4, %[[RESPTR]], %[[RESHOSTSIZE]])

  // Return ok status.
  // CHECK-NEXT: %[[OK:.+]] = emitc.call_opaque "iree_ok_status"()
  // CHECK-NEXT: return %[[OK]]
  vm.import private @variadic_fn(%arg0 : i32, %arg1 : i32 ...) -> i32

  vm.func @import_variadic(%arg0 : i32) -> i32 {
    %0 = vm.call.variadic @variadic_fn(%arg0, []) : (i32, i32 ...) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test vm.import conversion on a function with vm.ref arguments.
vm.module @my_module {

  // CHECK-LABEL: emitc.func private @my_module_call_0r_r_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
  // CHECK-SAME:                                        %arg2: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, %arg3: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {specifiers = ["static"]} {

  // Calculate the size of the arguments.
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.add %[[ARGSIZE0]], %[[ARGSIZE1]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.add %[[RESULTSIZE0]], %[[RESULTSIZE1]]
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"iree_host_size_t">

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "function"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.assign %[[ARGSTRUCTFN]] : !emitc.opaque<"iree_vm_function_t"> to %[[ARGSTRUCTFN_MEMBER]] : !emitc.opaque<"iree_vm_function_t">

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "arguments"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.apply "&"(%[[ARGBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[ARGSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[ARGSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSDATA:.+]] = "emitc.member_of_ptr"(%[[ARGBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGBYTESPANDATA]] : !emitc.ptr<ui8> to %[[ARGSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.apply "&"(%[[RESBYTESPAN_MEMBER]]) : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESSDATALENGTH:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data_length"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.assign %[[RESULTSIZE]] : !emitc.opaque<"iree_host_size_t"> to %[[RESSDATALENGTH]] : !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESSDATA:.+]] = "emitc.member_of_ptr"(%[[RESBYTESPAN]]) <{member = "data"}> : (!emitc.ptr<!emitc.opaque<"iree_byte_span_t">>) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESBYTESPANDATA]] : !emitc.ptr<ui8> to %[[RESSDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Pack the argument into the struct.
  // CHECK-NEXT: %[[ARGS:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGS_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "arguments"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: emitc.assign %[[ARGS_MEMBER]] : !emitc.opaque<"iree_byte_span_t"> to %[[ARGS]] : !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGSPTR_MEMBER:.+]] = "emitc.member"(%[[ARGS]]) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGSPTR_MEMBER]] : !emitc.ptr<ui8> to %[[ARGSPTR]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARG:.+]] = emitc.cast %[[ARGSPTR]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
  // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_assign"(%arg2, %[[ARG]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[MODULE_MEMBER:.+]] = "emitc.member_of_ptr"(%arg1) <{member = "module"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: emitc.assign %[[MODULE_MEMBER]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">> to %[[IMPORTMOD]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[BEGIN_CALL:.+]] = "emitc.member_of_ptr"(%[[IMPORTMOD]]) <{member = "begin_call"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_module_t">>) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_CALL_INDIRECT"(%[[BEGIN_CALL]], %[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RES_MEMBER:.+]] = "emitc.member"(%[[ARGSTRUCT]]) <{member = "results"}> : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: emitc.assign %[[RES_MEMBER]] : !emitc.opaque<"iree_byte_span_t"> to %[[RES]] : !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESPTR_MEMBER:.+]] = "emitc.member"(%[[RES]]) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESPTR_MEMBER]] : !emitc.ptr<ui8> to %[[RESPTR]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESREFPTR:.+]] = emitc.cast %[[RESPTR]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
  // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_move"(%[[RESREFPTR]], %arg3)

  // Return ok status.
  // CHECK-NEXT: %[[OK:.+]] = emitc.call_opaque "iree_ok_status"()
  // CHECK-NEXT: return %[[OK]]
  vm.import private @ref_fn(%arg0 : !vm.ref<?>) -> !vm.ref<?>

  vm.func @import_ref(%arg0 : !vm.ref<?>) -> !vm.ref<?> {
    %0 = vm.call @ref_fn(%arg0) : (!vm.ref<?>) -> !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }
}

// -----

vm.module @my_module {

  // Define structs for arguments and results
  //      CHECK: emitc.verbatim "struct my_module_fn_args_t {int32_t arg0;};"
  // CHECK-NEXT: emitc.verbatim "struct my_module_fn_result_t {int32_t res0;};"

  // Create a new function to export with the adapted signature.
  // CHECK:      emitc.func private @my_module_fn_export_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.opaque<"uint32_t">, %arg2: !emitc.opaque<"iree_byte_span_t">, %arg3: !emitc.opaque<"iree_byte_span_t">,
  // CHECK-SAME:                                %arg4: !emitc.ptr<!emitc.opaque<"void">>, %arg5: !emitc.ptr<!emitc.opaque<"void">>)
  // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">

  // Cast module and module state structs.
  // CHECK-NEXT: %[[MODULECASTED:.+]] = emitc.cast %arg4 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"struct my_module_t">>
  // CHECK-NEXT: %[[MODSTATECASTED:.+]] = emitc.cast %arg5 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"struct my_module_state_t">>

  // Cast argument and result structs.
  // CHECK-NEXT: %[[ARGDATA:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGDATA_MEMBER:.+]] = "emitc.member"(%arg2) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[ARGDATA_MEMBER]] : !emitc.ptr<ui8> to %[[ARGDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.cast %[[ARGDATA]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"struct my_module_fn_args_t">>
  // CHECK-NEXT: %[[RESULTDATA:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESULTDATA_MEMBER:.+]] = "emitc.member"(%arg3) <{member = "data"}> : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.assign %[[RESULTDATA_MEMBER]] : !emitc.ptr<ui8> to %[[RESULTDATA]] : !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESULTS:.+]] = emitc.cast %[[RESULTDATA]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"struct my_module_fn_result_t">>

  // Unpack the argument from the struct.
  // CHECK-NEXT: %[[MARG:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
  // CHECK-NEXT: %[[MARG_MEMBER:.+]] = "emitc.member_of_ptr"(%[[ARGS]]) <{member = "arg0"}> : (!emitc.ptr<!emitc.opaque<"struct my_module_fn_args_t">>) -> i32
  // CHECK-NEXT: emitc.assign %[[MARG_MEMBER]] : i32 to %[[MARG]] : i32

  // Unpack the result pointer from the struct.
  // CHECK-NEXT: %[[MRES_MEMBER:.+]] = "emitc.member_of_ptr"(%[[RESULTS]]) <{member = "res0"}> : (!emitc.ptr<!emitc.opaque<"struct my_module_fn_result_t">>) -> i32
  // CHECK-NEXT: %[[MRES:.+]] = emitc.apply "&"(%[[MRES_MEMBER]]) : (i32) -> !emitc.ptr<i32>

  // Call the internal function.
  // CHECK-NEXT: %{{.+}} = emitc.call @my_module_fn(%arg0, %[[MODULECASTED]], %[[MODSTATECASTED]], %[[MARG]], %[[MRES]])
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"struct my_module_t">>,
  // CHECK-SAME:        !emitc.ptr<!emitc.opaque<"struct my_module_state_t">>, i32, !emitc.ptr<i32>) -> !emitc.opaque<"iree_status_t">

  // Return ok status.
  // CHECK: %[[STATUS:.+]] = emitc.call_opaque "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
  // CHECK: return %[[STATUS]] : !emitc.opaque<"iree_status_t">

  vm.export @fn

  vm.func @fn(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_return
  vm.func @return(%arg0 : i32, %arg1 : !vm.ref<?>) -> (i32, !vm.ref<?>) {

    // Move the i32 value and ref into the result function arguments.
    // CHECK-NEXT: emitc.call_opaque "EMITC_DEREF_ASSIGN_VALUE"(%arg5, %arg3) : (!emitc.ptr<i32>, i32) -> ()

    // Create duplicate ref for
    // CHECK-NEXT: %[[REF:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[REFPTR:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK-NEXT: %[[REFSIZE:.+]] = emitc.call_opaque "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]} : () -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: emitc.call_opaque "memset"(%[[REFPTR]], %[[REFSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_host_size_t">) -> ()
    // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_move"(%arg4, %[[REFPTR]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()
    // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_move"(%[[REFPTR]], %arg6) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()

    // Release the ref.
    // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_release"(%arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()

    // Return ok status.
    // CHECK-NEXT: %[[STATUS:.+]] = emitc.call_opaque "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %[[STATUS]] : !emitc.opaque<"iree_status_t">
    vm.return %arg0, %arg1 : i32, !vm.ref<?>
  }
}

// -----

vm.module @my_module {
  vm.import private optional @optional_import_fn(%arg0 : i32) -> i32
  // CHECK-LABEL: emitc.func private @my_module_call_fn
  vm.func @call_fn() -> i32 {
    // CHECK-NEXT: %[[IMPORTS_VAR:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORTS:.+]] = "emitc.member_of_ptr"(%arg2) <{member = "imports"}> : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: emitc.assign %[[IMPORTS]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">> to %[[IMPORTS_VAR]] : !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS_VAR]]) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[MODULE:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
    // CHECK-NEXT: %[[MODULE_MEMBER:.+]] = "emitc.member_of_ptr"(%[[IMPORT]]) <{member = "module"}> : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
    // CHECK-NEXT: emitc.assign %[[MODULE_MEMBER]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">> to %[[MODULE]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
    // CHECK-NEXT: %[[CONDITION0:.+]] = emitc.logical_not %[[MODULE]] : !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
    // CHECK-NEXT: %[[CONDITION1:.+]] = emitc.logical_not %[[CONDITION0]] : i1
    // CHECK-NEXT: %[[RESULT:.+]] = emitc.cast %[[CONDITION1]] : i1 to i32
    %has_optional_import_fn = vm.import.resolved @optional_import_fn : i32
    vm.return %has_optional_import_fn : i32
  }
}
