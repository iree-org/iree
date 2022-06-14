// RUN: iree-opt --split-input-file --pass-pipeline="vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc)" %s | FileCheck %s

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
  vm.import @imported_fn(%arg0 : i32) -> i32

  // CHECK: func.func @my_module_call_imported_fn
  vm.func @call_imported_fn(%arg0 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // Create a variable for the function result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_call_[[IMPORTFN]](%arg0, %[[IMPORT]], %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>, i32, !emitc.ptr<!emitc.opaque<"int32_t">>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @imported_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
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
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_internal_fn(%arg0, %arg1, %arg2, %arg3, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"my_module_t">>,
    // CHECK-SAME:        !emitc.ptr<!emitc.opaque<"my_module_state_t">>, i32, !emitc.ptr<!emitc.opaque<"int32_t">>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @internal_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// Test vm.call.variadic conversion on an imported function.
vm.module @my_module {
  // CHECK: func.func @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import @variadic_fn(%arg0 : i32 ...) -> i32

  // CHECK: func.func @my_module_call_variadic
  vm.func @call_variadic(%arg0 : i32, %arg1 : i32) -> i32 {

    // Lookup import from module struct.
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]}
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>

    // This holds the number of variadic arguments.
    // CHECK-NEXT: %[[NARGS:.+]] = "emitc.constant"() {value = 2 : i32} : () -> i32

    // Create a variable for the result.
    // CHECK-NEXT: %[[RESULT:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> i32
    // CHECK-NEXT: %[[RESPTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>

    // Call the function created by the vm.import conversion.
    // CHECK-NEXT: %{{.+}} = call @my_module_call_[[VARIADICFN]](%arg0, %[[IMPORT]], %[[NARGS]], %arg3, %arg4, %[[RESPTR]])
    // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
    // CHECK-SAME:        i32, i32, i32, !emitc.ptr<!emitc.opaque<"int32_t">>)
    // CHECK-SAME:     -> !emitc.opaque<"iree_status_t">
    %0 = vm.call.variadic @variadic_fn([%arg0, %arg1]) : (i32 ...) -> i32
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
    // CHEKC: cf.br ^bb3
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

vm.module @my_module {
  // CHECK-LABEL: @my_module_fail
  vm.func @fail(%arg0 : i32) {

    // Typecast the argument to fail and branch respectively.
    // CHECK-NEXT: %[[COND:.+]] = emitc.cast %arg3 : i32 to i1
    // CHECK-NEXT: cf.cond_br %[[COND]], ^[[FAIL:.+]], ^[[OK:.+]]

    // In case of success, return ok status.
    // CHECK-NEXT: ^[[OK]]:
    // CHECK-NEXT: %[[OKSTATUS:.+]] = emitc.call "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %[[OKSTATUS]] : !emitc.opaque<"iree_status_t">

    // In case of fail, return status message.
    // CHECK-NEXT: ^[[FAIL]]:
    // CHECK-NEXT: %[[MSG:.+]] = emitc.call "iree_make_cstring_view"() {args = [#emitc.opaque<"\22message\22">]}
    // CHECK-SAME:     : () -> !emitc.opaque<"iree_string_view_t">
    // CHECK-NEXT: %[[MSGSIZE:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[MSG]]) {args = [0 : index, #emitc.opaque<"size">]}
    // CHECK-SAME:     : (!emitc.opaque<"iree_string_view_t">) -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: %[[MSGSIZEINT:.+]] = emitc.cast %[[MSGSIZE]] : !emitc.opaque<"iree_host_size_t"> to i32
    // CHECK-NEXT: %[[MSGDATA:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[MSG]]) {args = [0 : index, #emitc.opaque<"data">]}
    // CHECK-SAME:     : (!emitc.opaque<"iree_string_view_t">) -> !emitc.ptr<!emitc.opaque<"const char">>
    // CHECK-NEXT: %[[FAILSTATUS:.+]] = emitc.call "iree_status_allocate_f"(%[[MSGSIZEINT]], %[[MSGDATA]])
    // CHECK-SAME:     {args = [#emitc.opaque<"IREE_STATUS_FAILED_PRECONDITION">, #emitc.opaque<"\22<vm>\22">, 0 : i32, #emitc.opaque<"\22%.*s\22">, 0 : index, 1 : index]}
    // CHECK-SAME:     : (i32, !emitc.ptr<!emitc.opaque<"const char">>) -> !emitc.opaque<"iree_status_t">
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
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() {value = #emitc.opaque<"0">} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.call "EMITC_ADD"(%[[ARGSIZE0]], %[[ARGSIZE1]])

  // Calculate the size of the result. To avoid empty structs we insert a dummy value.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() {value = #emitc.opaque<"0">} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.call "EMITC_ADD"(%[[RESULTSIZE0]], %[[RESULTSIZE1]])

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.constant"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_MEMBER_ASSIGN"(%[[ARGSTRUCT]], %[[ARGSTRUCTFN]]) {args = [0 : index, #emitc.opaque<"function">, 1 : index]}

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.call "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : i32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.call "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESULTSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : i32, 1 : index]}

  // Check that we don't pack anything into the argument struct.
  // CHECK-NOT: emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-NOT: %[[ARGSPTR:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%{{.+}}) {args = [0 : index, #emitc.opaque<"data">]}

  // Create the call to the imported function.
  // CHECK-NEXT: %[[EXECRESULT:.+]] = "emitc.variable"() {value = #emitc.opaque<"">}
  // CHECK-SAME:     : () -> !emitc.opaque<"iree_vm_execution_result_t">
  // CHECK-NEXT: %[[EXECRESULTPTR1:.+]] = emitc.apply "&"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>
  // CHECK-NEXT: %[[EXECRESULTSIZE:.+]] = emitc.call "sizeof"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.call "memset"(%[[EXECRESULTPTR1]], %[[EXECRESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"module">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[ARGSTRUCTPTR:.+]] = emitc.apply "&"(%[[ARGSTRUCT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>
  // CHECK-NEXT: %[[EXECRESULTPTR:.+]] = emitc.apply "&"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>
  // CHECK-NEXT: %{{.+}} = emitc.call "EMITC_STRUCT_PTR_MEMBER_CALL"(%[[IMPORTMOD]], %arg0, %[[ARGSTRUCTPTR]], %[[EXECRESULTPTR]])
  // CHECK-SAME:     {args = [0 : index, #emitc.opaque<"begin_call">, 0 : index, 1 : index, 2 : index, 3 : index]}

  // Check that we don't unpack anything from the result struct.
  // CHECK-NOT: emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-NOT: emitc.call "EMITC_STRUCT_MEMBER"(%{{.+}}) {args = [0 : index, #emitc.opaque<"data">]}

  // Return ok status.
  //      CHECK: %[[OK:.+]] = emitc.call "iree_ok_status"()
  // CHECK-NEXT: return %[[OK]]
  vm.import @ref_fn() -> ()

  vm.func @import_ref() -> () {
    vm.call @ref_fn() : () -> ()
    vm.return
  }
}

// -----

// Test vm.import conversion on a variadic function.
vm.module @my_module {

  // CHECK-LABEL: func.func @my_module_call_0iCiD_i_2_import_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>,
  // CHECK-SAME:                                             %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: !emitc.ptr<!emitc.opaque<"int32_t">>)
  // CHECK-SAME:      -> !emitc.opaque<"iree_status_t"> attributes {emitc.static} {

  // Calculate the size of the arguments.
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() {value = #emitc.opaque<"0">} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]}
  // CHECK-NEXT: %[[ARGSIZE01:.+]] = emitc.call "EMITC_ADD"(%[[ARGSIZE0]], %[[ARGSIZE1]])
  // CHECK-NEXT: %[[ARGSIZE2:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]}
  // CHECK-NEXT: %[[ARGSIZE012:.+]] = emitc.call "EMITC_ADD"(%[[ARGSIZE01]], %[[ARGSIZE2]])
  // CHECK-NEXT: %[[ARGSIZE3:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]}
  // CHECK-NEXT: %[[ARGSIZE0123:.+]] = emitc.call "EMITC_ADD"(%[[ARGSIZE012]], %[[ARGSIZE3]])
  // CHECK-NEXT: %[[ARGSIZE4:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.call "EMITC_ADD"(%[[ARGSIZE0123]], %[[ARGSIZE4]])

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() {value = #emitc.opaque<"0">} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.call "EMITC_ADD"(%[[RESULTSIZE0]], %[[RESULTSIZE1]])

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.constant"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_MEMBER_ASSIGN"(%[[ARGSTRUCT]], %[[ARGSTRUCTFN]]) {args = [0 : index, #emitc.opaque<"function">, 1 : index]}

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.call "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call "iree_alloca"(%[[ARGSIZE]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : i32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.call "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call "iree_alloca"(%[[RESULTSIZE]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESULTSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : i32, 1 : index]}

  // Pack the arguments into the struct.
  // Here we also create pointers for non-pointer types.
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGS]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGHOSTSIZE:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1PTR:.+]] = emitc.apply "&"(%arg2) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>
  // CHECK-NEXT: emitc.call "memcpy"(%[[ARGSPTR]], %[[A1PTR]], %[[ARGHOSTSIZE]])
  // CHECK-NEXT: %[[ARGHOSTSIZE2:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A1ADDR:.+]] = emitc.call "EMITC_ADD"(%[[ARGSPTR]], %[[ARGHOSTSIZE2]])
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A1SIZE:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2PTR:.+]] = emitc.apply "&"(%arg3) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>
  // CHECK-NEXT: emitc.call "memcpy"(%[[A1ADDR]], %[[A2PTR]], %[[A1SIZE]])
  // CHECK-NEXT: %[[A1SIZE2:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A2ADDR:.+]] = emitc.call "EMITC_ADD"(%[[A1ADDR]], %[[A1SIZE2]])
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A2SIZE:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A3PTR:.+]] = emitc.apply "&"(%arg4) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>
  // CHECK-NEXT: emitc.call "memcpy"(%[[A2ADDR]], %[[A3PTR]], %[[A2SIZE]])
  // CHECK-NEXT: %[[A2SIZE2:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A3ADDR:.+]] = emitc.call "EMITC_ADD"(%[[A2ADDR]], %[[A2SIZE2]])
  // CHECK-SAME:     : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[A3SIZE:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[A4PTR:.+]] = emitc.apply "&"(%arg5) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>
  // CHECK-NEXT: emitc.call "memcpy"(%[[A3ADDR]], %[[A4PTR]], %[[A3SIZE:.+]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[EXECRESULT:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_execution_result_t">
  // CHECK-NEXT: %[[EXECRESULTPTR1:.+]] = emitc.apply "&"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>
  // CHECK-NEXT: %[[EXECRESULTSIZE:.+]] = emitc.call "sizeof"(%[[EXECRESULT]]) : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.call "memset"(%[[EXECRESULTPTR1]], %[[EXECRESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>, !emitc.opaque<"iree_host_size_t">) -> ()
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"module">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[ARGSTRUCTPTR:.+]] = emitc.apply "&"(%[[ARGSTRUCT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>
  // CHECK-NEXT: %[[EXECRESULTPTR:.+]] = emitc.apply "&"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>
  // CHECK-NEXT: %{{.+}} = emitc.call "EMITC_STRUCT_PTR_MEMBER_CALL"(%[[IMPORTMOD]], %arg0, %[[ARGSTRUCTPTR]], %[[EXECRESULTPTR]])
  // CHECK-SAME:     {args = [0 : index, #emitc.opaque<"begin_call">, 0 : index, 1 : index, 2 : index, 3 : index]}

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[RES]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESHOSTSIZE:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"int32_t">]} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.call "memcpy"(%arg6, %[[RESPTR]], %[[RESHOSTSIZE]])

  // Return ok status.
  // CHECK-NEXT: %[[OK:.+]] = emitc.call "iree_ok_status"()
  // CHECK-NEXT: return %[[OK]]
  vm.import @variadic_fn(%arg0 : i32, %arg1 : i32 ...) -> i32

  vm.func @import_variadic(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    %0 = vm.call.variadic @variadic_fn(%arg0, [%arg1, %arg2]) : (i32, i32 ...) -> i32
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
  // CHECK-NEXT: %[[ARGSIZE0:.+]] = "emitc.constant"() {value = #emitc.opaque<"0">} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[ARGSIZE1:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"iree_vm_ref_t">]}
  // CHECK-NEXT: %[[ARGSIZE:.+]] = emitc.call "EMITC_ADD"(%[[ARGSIZE0]], %[[ARGSIZE1]])

  // Calculate the size of the result.
  // CHECK-NEXT: %[[RESULTSIZE0:.+]] = "emitc.constant"() {value = #emitc.opaque<"0">} : () -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: %[[RESULTSIZE1:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"iree_vm_ref_t">]}
  // CHECK-NEXT: %[[RESULTSIZE:.+]] = emitc.call "EMITC_ADD"(%[[RESULTSIZE0]], %[[RESULTSIZE1]])

  // Create a struct for the arguments and results.
  // CHECK: %[[ARGSTRUCT:.+]] = "emitc.constant"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_function_call_t">
  // CHECK-NEXT: %[[ARGSTRUCTFN:.+]] = emitc.apply "*"(%arg1) : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.opaque<"iree_vm_function_t">
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_MEMBER_ASSIGN"(%[[ARGSTRUCT]], %[[ARGSTRUCTFN]]) {args = [0 : index, #emitc.opaque<"function">, 1 : index]}

  // Allocate space for the arguments.
  // CHECK-NEXT: %[[ARGBYTESPAN:.+]] = emitc.call "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[ARGBYTESPANDATAVOID:.+]] = emitc.call "iree_alloca"(%[[ARGSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[ARGBYTESPANDATA:.+]] = emitc.cast %[[ARGBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[ARGBYTESPAN]], %[[ARGBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call "memset"(%[[ARGBYTESPANDATA]], %[[ARGSIZE]]) {args = [0 : index, 0 : i32, 1 : index]}

  // Allocate space for the result.
  // CHECK-NEXT: %[[RESBYTESPAN:.+]] = emitc.call "EMITC_STRUCT_MEMBER_ADDRESS"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_byte_span_t">>
  // CHECK-NEXT: %[[RESBYTESPANDATAVOID:.+]] = emitc.call "iree_alloca"(%[[RESULTSIZE]]) : (!emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK-NEXT: %[[RESBYTESPANDATA:.+]] = emitc.cast %[[RESBYTESPANDATAVOID]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<ui8>
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESULTSIZE]]) {args = [0 : index, #emitc.opaque<"data_length">, 1 : index]}
  // CHECK-NEXT: emitc.call "EMITC_STRUCT_PTR_MEMBER_ASSIGN"(%[[RESBYTESPAN]], %[[RESBYTESPANDATA]]) {args = [0 : index, #emitc.opaque<"data">, 1 : index]}
  // CHECK-NEXT: emitc.call "memset"(%[[RESBYTESPANDATA]], %[[RESULTSIZE]]) {args = [0 : index, 0 : i32, 1 : index]}

  // Pack the argument into the struct.
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGSPTR:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGS]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARG:.+]] = emitc.cast %[[ARGSPTR]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
  // CHECK-NEXT: emitc.call "iree_vm_ref_assign"(%arg2, %[[ARG]])

  // Create the call to the imported function.
  // CHECK-NEXT: %[[EXECRESULT:.+]] = "emitc.variable"() {value = #emitc.opaque<"">}
  // CHECK-SAME:     : () -> !emitc.opaque<"iree_vm_execution_result_t">
  // CHECK-NEXT: %[[EXECRESULTPTR1:.+]] = emitc.apply "&"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>
  // CHECK-NEXT: %[[EXECRESULTSIZE:.+]] = emitc.call "sizeof"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.opaque<"iree_host_size_t">
  // CHECK-NEXT: emitc.call "memset"(%[[EXECRESULTPTR1]], %[[EXECRESULTSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]}
  // CHECK-NEXT: %[[IMPORTMOD:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"module">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
  // CHECK-NEXT: %[[ARGSTRUCTPTR:.+]] = emitc.apply "&"(%[[ARGSTRUCT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>
  // CHECK-NEXT: %[[EXECRESULTPTR:.+]] = emitc.apply "&"(%[[EXECRESULT]])
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_execution_result_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>
  // CHECK-NEXT: %{{.+}} = emitc.call "EMITC_STRUCT_PTR_MEMBER_CALL"(%[[IMPORTMOD]], %arg0, %[[ARGSTRUCTPTR]], %[[EXECRESULTPTR]])
  // CHECK-SAME:     {args = [0 : index, #emitc.opaque<"begin_call">, 0 : index, 1 : index, 2 : index, 3 : index]}

  // Unpack the function results.
  //      CHECK: %[[RES:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[ARGSTRUCT]]) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_vm_function_call_t">) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESPTR:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[RES]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESREFPTR:.+]] = emitc.cast %[[RESPTR]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
  // CHECK-NEXT: emitc.call "iree_vm_ref_move"(%[[RESREFPTR]], %arg3)

  // Return ok status.
  // CHECK-NEXT: %[[OK:.+]] = emitc.call "iree_ok_status"()
  // CHECK-NEXT: return %[[OK]]
  vm.import @ref_fn(%arg0 : !vm.ref<?>) -> !vm.ref<?>

  vm.func @import_ref(%arg0 : !vm.ref<?>) -> !vm.ref<?> {
    %0 = vm.call @ref_fn(%arg0) : (!vm.ref<?>) -> !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }
}

// -----

vm.module @my_module {

  // Typedef structs for arguments and results
  // CHECK: emitc.call "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_args_t">, #emitc.opaque<"int32_t arg0;">]} : () -> ()
  // CHECK: emitc.call "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_result_t">, #emitc.opaque<"int32_t res0;">]} : () -> ()

  // Create a new function to export with the adapted siganture.
  //      CHECK: func.func @my_module_fn_export_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.opaque<"iree_byte_span_t">, %arg2: !emitc.opaque<"iree_byte_span_t">,
  // CHECK-SAME:                                %arg3: !emitc.ptr<!emitc.opaque<"void">>, %arg4: !emitc.ptr<!emitc.opaque<"void">>, %arg5: !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>)
  // CHECK-SAME:     -> !emitc.opaque<"iree_status_t"> attributes {emitc.static, vm.calling_convention = "0i_i"}

  // Cast module and module state structs.
  // CHECK-NEXT: %[[MODULECASTED:.+]] = emitc.cast %arg3 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"my_module_t">>
  // CHECK-NEXT: %[[MODSTATECASTED:.+]] = emitc.cast %arg4 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"my_module_state_t">>

  // Cast argument and result structs.
  // CHECK-NEXT: %[[ARGDATA:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.cast %[[ARGDATA]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>
  // CHECK-NEXT: %[[RESULTDATA:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESULTS:.+]] = emitc.cast %[[RESULTDATA]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>

  // Unpack the argument from the struct.
  // CHECK-NEXT: %[[MARG:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%[[ARGS]]) {args = [0 : index, #emitc.opaque<"arg0">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>) -> i32

  // Unpack the result pointer from the struct.
  // CHECK-NEXT: %[[MRES:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER_ADDRESS"(%[[RESULTS]]) {args = [0 : index, #emitc.opaque<"res0">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>) -> !emitc.ptr<!emitc.opaque<"int32_t">>

  // Call the internal function.
  // CHECK-NEXT: %{{.+}} = call @my_module_fn(%arg0, %[[MODULECASTED]], %[[MODSTATECASTED]], %[[MARG]], %[[MRES]])
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"my_module_t">>,
  // CHECK-SAME:        !emitc.ptr<!emitc.opaque<"my_module_state_t">>, i32, !emitc.ptr<!emitc.opaque<"int32_t">>) -> !emitc.opaque<"iree_status_t">

  // Return ok status.
  // CHECK: %[[STATUS:.+]] = emitc.call "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
  // CHECK: return %[[STATUS]] : !emitc.opaque<"iree_status_t">

  // Export the new function.
  // CHECK: vm.export @my_module_fn_export_shim as("fn") attributes {ordinal = 0 : i32}
  vm.export @fn

  vm.func @fn(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_return
  vm.func @return(%arg0 : i32, %arg1 : !vm.ref<?>) -> (i32, !vm.ref<?>) {

    // Create duplicate ref for
    // CHECK-NEXT: %[[REF:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[REFPTR:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK-NEXT: %[[REFSIZE:.+]] = emitc.call "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]} : () -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: emitc.call "memset"(%[[REFPTR]], %[[REFSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_host_size_t">) -> ()
    // CHECK-NEXT: emitc.call "iree_vm_ref_move"(%arg4, %[[REFPTR]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()

    // Move the i32 value and ref into the result function arguments.
    // CHECK-NEXT: emitc.call "EMITC_DEREF_ASSIGN_VALUE"(%arg5, %arg3) : (!emitc.ptr<!emitc.opaque<"int32_t">>, i32) -> ()
    // CHECK-NEXT: emitc.call "iree_vm_ref_move"(%[[REFPTR]], %arg6) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()

    // Release the ref.
    // CHECK-NEXT: emitc.call "iree_vm_ref_release"(%arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()

    // Return ok status.
    // CHECK-NEXT: %[[STATUS:.+]] = emitc.call "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %[[STATUS]] : !emitc.opaque<"iree_status_t">
    vm.return %arg0, %arg1 : i32, !vm.ref<?>
  }
}

// -----

vm.module @my_module {
  vm.import optional @optional_import_fn(%arg0 : i32) -> i32
  // CHECK-LABEL: @my_module_call_fn
  vm.func @call_fn() -> i32 {
    // CHECK-NEXT: %[[IMPORTS:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[IMPORT:.+]] = emitc.call "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[IMPORTS]]) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %[[MODULE:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%[[IMPORT]]) {args = [0 : index, #emitc.opaque<"module">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_module_t">>
    // CHECK-NEXT: %[[CONDITION0:.+]] = emitc.call "EMITC_NOT"(%[[MODULE]]) {args = [0 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_module_t">>) -> i1
    // CHECK-NEXT: %[[CONDITION1:.+]] = emitc.call "EMITC_NOT"(%[[CONDITION0]]) {args = [0 : index]} : (i1) -> i1
    // CHECK-NEXT: %[[RESULT:.+]] = emitc.cast %[[CONDITION1]] : i1 to i32
    %has_optional_import_fn = vm.import.resolved @optional_import_fn : i32
    vm.return %has_optional_import_fn : i32
  }
}
