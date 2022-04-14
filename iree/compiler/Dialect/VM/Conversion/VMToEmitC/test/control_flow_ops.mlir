// RUN: iree-opt -split-input-file -pass-pipeline="vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc)" %s | FileCheck %s

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
  // CHECK: func @my_module_call_[[IMPORTFN:[^\(]+]]
  vm.import @imported_fn(%arg0 : i32) -> i32

  // CHECK: func @my_module_call_imported_fn
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
  // CHECK: func @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import @variadic_fn(%arg0 : i32 ...) -> i32

  // CHECK: func @my_module_call_variadic
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
    // CHECK-NEXT: %[[COND:.+]] = emitc.call "EMITC_CAST"(%arg3) {args = [0 : index, i1]} : (i32) -> i1
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
    // CHECK-NEXT: %[[MSGSIZEINT:.+]] = emitc.call "EMITC_CAST"(%[[MSGSIZE]]) {args = [0 : index, #emitc.opaque<"int">]}
    // CHECK-SAME:     : (!emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"int">
    // CHECK-NEXT: %[[MSGDATA:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[MSG]]) {args = [0 : index, #emitc.opaque<"data">]}
    // CHECK-SAME:     : (!emitc.opaque<"iree_string_view_t">) -> !emitc.ptr<!emitc.opaque<"const char">>
    // CHECK-NEXT: %[[FAILSTATUS:.+]] = emitc.call "iree_status_allocate_f"(%[[MSGSIZEINT]], %[[MSGDATA]])
    // CHECK-SAME:     {args = [#emitc.opaque<"IREE_STATUS_FAILED_PRECONDITION">, #emitc.opaque<"\22<vm>\22">, 0 : i32, #emitc.opaque<"\22%.*s\22">, 0 : index, 1 : index]}
    // CHECK-SAME:     : (!emitc.opaque<"int">, !emitc.ptr<!emitc.opaque<"const char">>) -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %[[FAILSTATUS]] : !emitc.opaque<"iree_status_t">
    vm.fail %arg0, "message"
  }
}

// -----

vm.module @my_module {

  // Typedef structs for arguments and results
  // CHECK: emitc.call "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_args_t">, #emitc.opaque<"int32_t arg0;">]} : () -> ()
  // CHECK: emitc.call "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_result_t">, #emitc.opaque<"int32_t res0;">]} : () -> ()

  // Create a new function to export with the adapted siganture.
  //      CHECK: func @my_module_fn_export_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>,
  // CHECK-SAME:                                %arg2: !emitc.ptr<!emitc.opaque<"void">>, %arg3: !emitc.ptr<!emitc.opaque<"void">>, %arg4: !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>)
  // CHECK-SAME:     -> !emitc.opaque<"iree_status_t"> attributes {emitc.static, vm.calling_convention = "0i_i"}

  // Cast module and module state structs.
  // CHECK-NEXT: %[[MODULECASTED:.+]] = emitc.call "EMITC_CAST"(%arg2) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_t">>]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"my_module_t">>
  // CHECK-NEXT: %[[MODSTATECASTED:.+]] = emitc.call "EMITC_CAST"(%arg3) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_state_t">>]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"my_module_state_t">>

  // Cast argument and result structs.
  // CHECK-NEXT: %[[CALLARGS:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"arguments">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[ARGDATA:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[CALLARGS]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[ARGS:.+]] = emitc.call "EMITC_CAST"(%[[ARGDATA]]) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>]}
  // CHECK-SAME:     : (!emitc.ptr<ui8>) -> !emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>
  // CHECK-NEXT: %[[CALLRESULT:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"results">]}
  // CHECK-SAME:     : (!emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %[[RESULTDATA:.+]] = emitc.call "EMITC_STRUCT_MEMBER"(%[[CALLRESULT]]) {args = [0 : index, #emitc.opaque<"data">]}
  // CHECK-SAME:     : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %[[RESULTS:.+]] = emitc.call "EMITC_CAST"(%[[RESULTDATA]]) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>]}
  // CHECK-SAME:     : (!emitc.ptr<ui8>) -> !emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>

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
    // CHECK-NEXT: %[[REFSIZE:.+]] = emitc.call "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]} : () -> i32
    // CHECK-NEXT: emitc.call "memset"(%[[REFPTR]], %[[REFSIZE]]) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, i32) -> ()
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
