// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @my_module_branch_empty
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
  // CHECK-LABEL: @my_module_branch_int_args
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
  // CHECK-LABEL: @my_module_branch_ref_args
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
  // CHECK-LABEL: @my_module_branch_mixed_args
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
  // CHECK: func.func @my_module_call_[[IMPORTFN:[^\(]+]]
  vm.import private @imported_fn(%arg0 : i32) -> i32

  // CHECK: func.func @my_module_call_imported_fn
  vm.func @call_imported_fn(%arg0 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // Create a variable for the function result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_call_[[IMPORTFN]](%arg0, %[[IMPORT]], %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>, i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @imported_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test that the order of imports and calls doesn't matter.
vm.module @my_module {
  // CHECK: func.func @my_module_call_imported_fn
  vm.func @call_imported_fn(%arg0 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // Create a variable for the function result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_call_[[IMPORTFN:[^\(]+]](%arg0, %[[IMPORT]], %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>, i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @imported_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }

  // CHECK: func.func @my_module_call_[[IMPORTFN]]
  vm.import private @imported_fn(%arg0 : i32) -> i32
}

// -----

// Test vm.call conversion on an internal function.
vm.module @my_module {
  vm.func @internal_fn(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }

  // CHECK-LABEL: @my_module_call_internal_fn
  vm.func @call_internal_fn(%arg0 : i32) -> i32 {

    // Create a variable for the result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_internal_fn(%arg0, %arg1, %arg2, %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"my_module_t">>,
    // CHECK-SAME:        !emitc.ptr<!emitc.opaque<"my_module_state_t">>, i32, !emitc.ptr<i32>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @internal_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test vm.call.variadic conversion on an imported function.
vm.module @my_module {
  // CHECK: func.func @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import private @variadic_fn(%arg0 : i32 ...) -> i32

  // CHECK: func.func @my_module_call_variadic
  vm.func @call_variadic(%arg0 : i32, %arg1 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // This holds the number of variadic arguments.
    // CHECK-NEXT: %[[NARGS:.+]] = "emitc.constant"() <{value = 2 : i32}> : () -> i32

    // Create a variable for the result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_call_[[VARIADICFN]](%arg0, %[[IMPORT]], %[[NARGS]], %arg3, %arg4, %[[RESPTR]])
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
  // CHECK: func.func @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import private @variadic_fn(%arg0 : i32 ...) -> i32

  // CHECK: func.func @my_module_call_variadic
  vm.func @call_variadic() -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // This holds the number of variadic arguments.
    // CHECK-NEXT: %[[NARGS:.+]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32

    // Create a variable for the result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_call_[[VARIADICFN]](%arg0, %[[IMPORT]], %[[NARGS]], %[[RESPTR]])
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
  // CHECK: func.func @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import private @variadic_fn(%is : i32 ..., %fs : f32 ...) -> i32

  // CHECK: func.func @my_module_call_variadic
  vm.func @call_variadic(%i : i32, %f : f32) -> i32 {

    %0 = vm.call.variadic @variadic_fn([%i, %i], [%f, %f, %f]) : (i32 ..., f32 ...) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_cond_branch_empty
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
  // CHECK-LABEL: @my_module_cond_branch_int_args
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
  // CHECK-LABEL: @my_module_cond_branch_ref_args
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

// CHECK-LABEL: @my_module_br_table_empty
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

// CHECK-LABEL: @my_module_br_table
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
  // CHECK-LABEL: @my_module_fail
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
    // CHECK-NEXT: %[[MSGSIZE:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[MSG]]) {args = [0 : index, #emitc.opaque<"size">]}
    // CHECK-SAME:     : (!emitc.opaque<"iree_string_view_t">) -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: %[[MSGSIZEINT:.+]] = emitc.cast %[[MSGSIZE]] : !emitc.opaque<"iree_host_size_t"> to !emitc.opaque<"int">
    // CHECK-NEXT: %[[MSGDATA:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[MSG]]) {args = [0 : index, #emitc.opaque<"data">]}
    // CHECK-SAME:     : (!emitc.opaque<"iree_string_view_t">) -> !emitc.ptr<!emitc.opaque<"const char">>
    // CHECK-NEXT: %[[FAILSTATUS:.+]] = emitc.call_opaque "iree_status_allocate_f"(%[[MSGSIZEINT]], %[[MSGDATA]])
    // CHECK-SAME:     {args = [#emitc.opaque<"IREE_STATUS_FAILED_PRECONDITION">, #emitc.opaque<"\22<vm>\22">, 0 : i32, #emitc.opaque<"\22%.*s\22">, 0 : index, 1 : index]}
    // CHECK-SAME:     : (!emitc.opaque<"int">, !emitc.ptr<!emitc.opaque<"const char">>) -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %[[FAILSTATUS]] : !emitc.opaque<"iree_status_t">
    vm.fail %arg0, "message"
  }
}

// -----

// Test vm.import conversion on a void function.
vm.module @my_module {

  // CHECK-LABEL: func.func @my_module_call_0v_v_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {emitc.static} {

  // Calculate the size of the arguments. To avoid empty structs we insert a dummy value.
  // CHECK-NEXT: %[[ARGSIZE:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">

  // Calculate the size of the result. To avoid empty structs we insert a dummy value.
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_MEMBER_ASSIGN"(%[[ARGSTRUCT]], %[[ARGSTRUCTFN]]) {args = [0 : index, #emitc.opaque<"function">, 1 : index]}

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESULTSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Check that we don't pack anything into the argument struct.
  // CHECK-NOT: emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-NOT: %[[ARGSPTR:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%{{.+}}) {args = [0 : index, #emitc.opaque<"data">]}

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"module">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_CALL"(%[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])
  // CHECK-SAME:     {args = [0 : index, #emitc.opaque<"begin_call">, 0 : index, 1 : index, 2 : index]}

  // Check that we don't unpack anything from the result struct.
  // CHECK-NOT: emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-NOT: emitc.call_opaque "EMITC_STRUCT_MEMBER"(%{{.+}}) {args = [0 : index, #emitc.opaque<"data">]}

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

  // CHECK-LABEL: func.func @my_module_call_0iCiD_i_2_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
  // CHECK-SAME:                                             %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: !emitc.ptr<i32>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {emitc.static} {

  // Calculate the size of the arguments.
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE01:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSIZE0]], %[[ARGSIZE1]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-NEXT: %[[ARGSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE012:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSIZE01]], %[[ARGSIZE2]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-NEXT: %[[ARGSIZE3:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE0123:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSIZE012]], %[[ARGSIZE3]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-NEXT: %[[ARGSIZE4:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSIZE0123]], %[[ARGSIZE4]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[RESULTSIZE0]], %[[RESULTSIZE1]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_MEMBER_ASSIGN"(%[[ARGSTRUCT]], %[[ARGSTRUCTFN]]) {args = [0 : index, #emitc.opaque<"function">, 1 : index]}

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESULTSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Pack the arguments into the struct.
  // Here we also create pointers for non-pointer types.
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGS]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGHOSTSIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1PTR:.+]] = emitc.apply "&"(%arg2) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[ARGSPTR]], %[[A1PTR]], %[[ARGHOSTSIZE]])
  // CHECK-NEXT: %[[ARGHOSTSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1ADDR:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSPTR]], %[[ARGHOSTSIZE2]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A1SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2PTR:.+]] = emitc.apply "&"(%arg3) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A1ADDR]], %[[A2PTR]], %[[A1SIZE]])
  // CHECK-NEXT: %[[A1SIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2ADDR:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[A1ADDR]], %[[A1SIZE2]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A2SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A3PTR:.+]] = emitc.apply "&"(%arg4) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A2ADDR]], %[[A3PTR]], %[[A2SIZE]])
  // CHECK-NEXT: %[[A2SIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A3ADDR:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[A2ADDR]], %[[A2SIZE2]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A3SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A4PTR:.+]] = emitc.apply "&"(%arg5) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A3ADDR]], %[[A4PTR]], %[[A3SIZE:.+]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"module">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_CALL"(%[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])
  // CHECK-SAME:     {args = [0 : index, #emitc.opaque<"begin_call">, 0 : index, 1 : index, 2 : index]}

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[RES]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
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

  // CHECK-LABEL: func.func @my_module_call_0iCiD_i_0_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
  // CHECK-SAME:                                             %arg2: i32, %arg3: i32, %arg4: !emitc.ptr<i32>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {emitc.static} {

  // Calculate the size of the arguments.
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE01:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSIZE0]], %[[ARGSIZE1]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-NEXT: %[[ARGSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSIZE01]], %[[ARGSIZE2]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [i32]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[RESULTSIZE0]], %[[RESULTSIZE1]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_MEMBER_ASSIGN"(%[[ARGSTRUCT]], %[[ARGSTRUCTFN]]) {args = [0 : index, #emitc.opaque<"function">, 1 : index]}

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESULTSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Pack the arguments into the struct.
  // Here we also create pointers for non-pointer types.
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGS]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGHOSTSIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1PTR:.+]] = emitc.apply "&"(%arg2) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[ARGSPTR]], %[[A1PTR]], %[[ARGHOSTSIZE]])
  // CHECK-NEXT: %[[ARGHOSTSIZE2:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1ADDR:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSPTR]], %[[ARGHOSTSIZE2]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A1SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2PTR:.+]] = emitc.apply "&"(%arg3) : (i32) -> !emitc.ptr<i32>
  // CHECK-NEXT: emitc.call_opaque "memcpy"(%[[A1ADDR]], %[[A2PTR]], %[[A1SIZE]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"module">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_CALL"(%[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])
  // CHECK-SAME:     {args = [0 : index, #emitc.opaque<"begin_call">, 0 : index, 1 : index, 2 : index]}

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[RES]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
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

  // CHECK-LABEL: func.func @my_module_call_0r_r_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
  // CHECK-SAME:                                        %arg2: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, %arg3: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {emitc.static} {

  // Calculate the size of the arguments.
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[ARGSIZE0]], %[[ARGSIZE1]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call_opaque "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.call_opaque "EMITC_BINARY"(%[[RESULTSIZE0]], %[[RESULTSIZE1]]) {args = [#emitc.opaque<"+">, 0 : index, 1 : index]}

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_MEMBER_ASSIGN"(%[[ARGSTRUCT]], %[[ARGSTRUCTFN]]) {args = [0 : index, #emitc.opaque<"function">, 1 : index]}

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call_opaque "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESULTSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call_opaque "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}

  // Pack the argument into the struct.
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGS]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARG:.+]] = emitc.cast %[[ARGSPTR]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
  // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_assign"(%arg2, %[[ARG]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"module">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %{{.+}} = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_CALL"(%[[IMPORTMOD]], %arg0, %[[ARGSTRUCT]])
  // CHECK-SAME:     {args = [0 : index, #emitc.opaque<"begin_call">, 0 : index, 1 : index, 2 : index]}

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%[[RES]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
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

  // Typedef structs for arguments and results
  // CHECK: emitc.call_opaque "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_args_t">, #emitc.opaque<"int32_t arg0;">]} : () -> ()
  // CHECK: emitc.call_opaque "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_result_t">, #emitc.opaque<"int32_t res0;">]} : () -> ()

  // Create a new function to export with the adapted signature.
  //      CHECK: func.func @my_module_fn_export_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.opaque<"uint32_t">, %arg2: !emitc.opaque<"iree_byte_span_t">, %arg3: !emitc.opaque<"iree_byte_span_t">,
  // CHECK-SAME:                                %arg4: !emitc.ptr<!emitc.opaque<"void">>, %arg5: !emitc.ptr<!emitc.opaque<"void">>)
  // CHECK-SAME:     -> !emitc.opaque<"iree_status_t"> attributes {emitc.static, vm.calling_convention = "0i_i", vm.export_name = "fn"}

  // Cast module and module state structs.
  // CHECK-NEXT: %[[MODULECASTED:.+]] = emitc.cast %arg4 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"my_module_t">>
  // CHECK-NEXT: %[[MODSTATECASTED:.+]] = emitc.cast %arg5 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"my_module_state_t">>

  // Cast argument and result structs.
  // CHECK-NEXT: %[[ARGDATA:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.cast %[[ARGDATA]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>
  // CHECK-NEXT: %[[RESULTDATA:.+]] = emitc.call_opaque "EMITC_STRUCT_MEMBER"(%arg3) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESULTS:.+]] = emitc.cast %[[RESULTDATA]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>

  // Unpack the argument from the struct.
  // CHECK-NEXT: %[[MARG:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%[[ARGS]]) {args = [0 : index, #emitc.opaque<"arg0">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>) -> i32

  // Unpack the result pointer from the struct.
  // CHECK-NEXT: %[[MRES:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER_ADDRESS"(%[[RESULTS]]) {args = [0 : index, #emitc.opaque<"res0">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>) -> !emitc.ptr<i32>

  // Call the internal function.
  // CHECK-NEXT: %{{.+}} = call @my_module_fn(%arg0, %[[MODULECASTED]], %[[MODSTATECASTED]], %[[MARG]], %[[MRES]])
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"my_module_t">>,
  // CHECK-SAME:        !emitc.ptr<!emitc.opaque<"my_module_state_t">>, i32, !emitc.ptr<i32>) -> !emitc.opaque<"iree_status_t">

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
  // CHECK-LABEL: @my_module_return
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
  // CHECK-LABEL: @my_module_call_fn
  vm.func @call_fn() -> i32 {
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call_opaque "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[MODULE:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%[[IMPORT]]) {args = [0 : index, #emitc.opaque<"module">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
    // CHECK-NEXT: %[[CONDITION0:.+]] = emitc.call_opaque "EMITC_UNARY"(%[[MODULE]]) {args = [#emitc.opaque<"!">, 0 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_module_t">>) -> i1
    // CHECK-NEXT: %[[CONDITION1:.+]] = emitc.call_opaque "EMITC_UNARY"(%[[CONDITION0]]) {args = [#emitc.opaque<"!">, 0 : index]} : (i1) -> i1
    // CHECK-NEXT: %[[RESULT:.+]] = emitc.cast %[[CONDITION1]] : i1 to i32
    %has_optional_import_fn = vm.import.resolved @optional_import_fn : i32
    vm.return %has_optional_import_fn : i32
  }
}
