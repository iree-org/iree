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

// TODO: different operands
vm.module @my_module {
  // CHECK: func @my_module_call_[[IMPORTFN:[^\(]+]]
  vm.import @imported_fn(%arg0 : i32) -> i32

  // CHECK: func @my_module_call_imported_fn
  vm.func @call_imported_fn(%arg0 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %1 = emitc.call "EMITC_ARRAY_ELEMENT_ADDRESS"(%0) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %2 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> i32
    // CHECK-NEXT: %3 = emitc.apply "&"(%2) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>
    // CHECK-NEXT: %4 = call @my_module_call_[[IMPORTFN]](%arg0, %1, %arg3, %3) : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>, i32, !emitc.ptr<!emitc.opaque<"int32_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @imported_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  vm.func @internal_fn(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }

  // CHECK-LABEL: @my_module_call_internal_fn
  vm.func @call_internal_fn(%arg0 : i32) -> i32 {
    // CHECK-NEXT: %0 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> i32
    // CHECK-NEXT: %1 = emitc.apply "&"(%0) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>
    // CHECK-NEXT: %2 = call @my_module_internal_fn(%arg0, %arg1, %arg2, %arg3, %1) : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"my_module_t">>, !emitc.ptr<!emitc.opaque<"my_module_state_t">>, i32, !emitc.ptr<!emitc.opaque<"int32_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.call @internal_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK: func @my_module_call_[[VARIADICFN:[^\(]+]]
  vm.import @variadic_fn(%arg0 : i32 ...) -> i32

  // CHECK: func @my_module_call_variadic
  vm.func @call_variadic(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"imports">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %1 = emitc.call "EMITC_ARRAY_ELEMENT_ADDRESS"(%0) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>
    // CHECK-NEXT: %2 = "emitc.constant"() {value = 2 : i32} : () -> i32
    // CHECK-NEXT: %3 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> i32
    // CHECK-NEXT: %4 = emitc.apply "&"(%3) : (i32) -> !emitc.ptr<!emitc.opaque<"int32_t">>
    // CHECK-NEXT: %5 = call @my_module_call_[[VARIADICFN]](%arg0, %1, %2, %arg3, %arg4, %4) : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_function_t">>, i32, i32, i32, !emitc.ptr<!emitc.opaque<"int32_t">>) -> !emitc.opaque<"iree_status_t">
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
    // CHECK-NEXT: %0 = emitc.call "EMITC_CAST"(%arg3) {args = [0 : index, i1]} : (i32) -> i1
    // CHECK-NEXT: cf.cond_br %0, ^bb2, ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: %1 = emitc.call "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %1 : !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: %2 = emitc.call "iree_make_cstring_view"() {args = [#emitc.opaque<"\22message\22">]} : () -> !emitc.opaque<"iree_string_view_t">
    // CHECK-NEXT: %3 = emitc.call "EMITC_STRUCT_MEMBER"(%2) {args = [0 : index, #emitc.opaque<"size">]} : (!emitc.opaque<"iree_string_view_t">) -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: %4 = emitc.call "EMITC_CAST"(%3) {args = [0 : index, #emitc.opaque<"int">]} : (!emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"int">
    // CHECK-NEXT: %5 = emitc.call "EMITC_STRUCT_MEMBER"(%2) {args = [0 : index, #emitc.opaque<"data">]} : (!emitc.opaque<"iree_string_view_t">) -> !emitc.ptr<!emitc.opaque<"const char">>
    // CHECK-NEXT: %6 = emitc.call "iree_status_allocate_f"(%4, %5) {args = [#emitc.opaque<"IREE_STATUS_FAILED_PRECONDITION">, #emitc.opaque<"\22<vm>\22">, 0 : i32, #emitc.opaque<"\22%.*s\22">, 0 : index, 1 : index]} : (!emitc.opaque<"int">, !emitc.ptr<!emitc.opaque<"const char">>) -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %6 : !emitc.opaque<"iree_status_t">
    vm.fail %arg0, "message"
  }
}

// -----

vm.module @my_module {
  // CHECK: emitc.call "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_args_t">, #emitc.opaque<"int32_t arg0;">]} : () -> ()
  // CHECK: emitc.call "EMITC_TYPEDEF_STRUCT"() {args = [#emitc.opaque<"my_module_fn_result_t">, #emitc.opaque<"int32_t res0;">]} : () -> ()
  // CHECK: func @my_module_fn_export_shim(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, %arg1: !emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>, %arg2: !emitc.ptr<!emitc.opaque<"void">>, %arg3: !emitc.ptr<!emitc.opaque<"void">>, %arg4: !emitc.ptr<!emitc.opaque<"iree_vm_execution_result_t">>) -> !emitc.opaque<"iree_status_t"> attributes {emitc.static, vm.calling_convention = "0i_i"} {
  // CHECK-NEXT: %0 = emitc.call "EMITC_CAST"(%arg2) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_t">>]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"my_module_t">>
  // CHECK-NEXT: %1 = emitc.call "EMITC_CAST"(%arg3) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_state_t">>]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"my_module_state_t">>
  // CHECK-NEXT: %2 = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"arguments">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %3 = emitc.call "EMITC_STRUCT_MEMBER"(%2) {args = [0 : index, #emitc.opaque<"data">]} : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %4 = emitc.call "EMITC_CAST"(%3) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>]} : (!emitc.ptr<ui8>) -> !emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>
  // CHECK-NEXT: %5 = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg1) {args = [0 : index, #emitc.opaque<"results">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_function_call_t">>) -> !emitc.opaque<"iree_byte_span_t">
  // CHECK-NEXT: %6 = emitc.call "EMITC_STRUCT_MEMBER"(%5) {args = [0 : index, #emitc.opaque<"data">]} : (!emitc.opaque<"iree_byte_span_t">) -> !emitc.ptr<ui8>
  // CHECK-NEXT: %7 = emitc.call "EMITC_CAST"(%6) {args = [0 : index, !emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>]} : (!emitc.ptr<ui8>) -> !emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>
  // CHECK-NEXT: %8 = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%4) {args = [0 : index, #emitc.opaque<"arg0">]} : (!emitc.ptr<!emitc.opaque<"my_module_fn_args_t">>) -> i32
  // CHECK-NEXT: %9 = emitc.call "EMITC_STRUCT_PTR_MEMBER_ADDRESS"(%7) {args = [0 : index, #emitc.opaque<"res0">]} : (!emitc.ptr<!emitc.opaque<"my_module_fn_result_t">>) -> !emitc.ptr<!emitc.opaque<"int32_t">>
  // CHECK-NEXT: %10 = call @my_module_fn(%arg0, %0, %1, %8, %9) : (!emitc.ptr<!emitc.opaque<"iree_vm_stack_t">>, !emitc.ptr<!emitc.opaque<"my_module_t">>, !emitc.ptr<!emitc.opaque<"my_module_state_t">>, i32, !emitc.ptr<!emitc.opaque<"int32_t">>) -> !emitc.opaque<"iree_status_t">
  // CHECK: %[[STATUS:.+]] = emitc.call "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
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
    // CHECK-NEXT: %0 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.apply "&"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK-NEXT: %2 = emitc.call "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]} : () -> i32
    // CHECK-NEXT: emitc.call "memset"(%1, %2) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, i32) -> ()
    // CHECK-NEXT: emitc.call "iree_vm_ref_move"(%arg4, %1) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()
    // CHECK-NEXT: emitc.call "EMITC_DEREF_ASSIGN_VALUE"(%arg5, %arg3) : (!emitc.ptr<!emitc.opaque<"int32_t">>, i32) -> ()
    // CHECK-NEXT: emitc.call "iree_vm_ref_move"(%1, %arg6) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()
    // CHECK-NEXT: emitc.call "iree_vm_ref_release"(%arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()
    // CHECK-NEXT: %3 = emitc.call "iree_ok_status"() : () -> !emitc.opaque<"iree_status_t">
    // CHECK-NEXT: return %3 : !emitc.opaque<"iree_status_t">
    vm.return %arg0, %arg1 : i32, !vm.ref<?>
  }
}
