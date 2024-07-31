// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_alloc
  vm.func @list_alloc(%arg0: i32) -> !vm.list<i32> {
    // CHECK: %[[LIST_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>>
    // CHECK: %[[LIST_PTR:.+]] = emitc.apply "&"(%3) : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>>) -> !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>>
    // CHECK-DAG: %[[STATE_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-DAG: emitc.assign %arg2 : !emitc.ptr<!emitc.opaque<"struct my_module_state_t">> to %[[STATE_LVAL]] : <!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-DAG: %[[ALLOCATOR_LVAL:.+]] = "emitc.member_of_ptr"(%[[STATE_LVAL]]) <{member = "allocator"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>) -> !emitc.lvalue<!emitc.opaque<"iree_allocator_t">>
    // CHECK-DAG: %[[ALLOCATOR:.+]] = emitc.load %[[ALLOCATOR_LVAL]] : <!emitc.opaque<"iree_allocator_t">>

    // CHECK: %[[TYPE_DEF:.+]] = emitc.call_opaque "iree_vm_make_value_type_def"() {args = [#emitc.opaque<"IREE_VM_VALUE_TYPE_I32">]} : () -> !emitc.opaque<"iree_vm_type_def_t">
    // CHECK-NEXT: %[[STATUS:.+]] = emitc.call_opaque "iree_vm_list_create"(%[[TYPE_DEF]], %arg3, %[[ALLOCATOR]], %[[LIST_PTR]]) : (!emitc.opaque<"iree_vm_type_def_t">, i32, !emitc.opaque<"iree_allocator_t">, !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>>) -> !emitc.opaque<"iree_status_t">

    // CHECK: %[[LIST_TYPE_ID:.+]] = emitc.call_opaque "iree_vm_list_type"() : () -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK-NEXT: %[[LIST:.+]] = emitc.load %[[LIST_LVAL]] : <!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>>
    // CHECK-NEXT: %[[STATUS2:.+]] = emitc.call_opaque "iree_vm_ref_wrap_assign"(%[[LIST]], %[[LIST_TYPE_ID]], %arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, !emitc.opaque<"iree_vm_ref_type_t">, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">

    %0 = vm.list.alloc %arg0 : (i32) -> !vm.list<i32>
    vm.return %0 : !vm.list<i32>
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_reserve
  vm.func @list_reserve(%arg0: !vm.list<i32>, %arg1: i32) {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call_opaque "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_reserve"(%1, %arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32) -> !emitc.opaque<"iree_status_t">
    vm.list.reserve %arg0, %arg1 : (!vm.list<i32>, i32)
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_resize
  vm.func @list_resize(%arg0: !vm.list<i32>, %arg1: i32) {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call_opaque "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_resize"(%1, %arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32) -> !emitc.opaque<"iree_status_t">
    vm.list.resize %arg0, %arg1 : (!vm.list<i32>, i32)
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_size
  vm.func @list_size(%arg0: !vm.list<i32>) -> i32 {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call_opaque "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_size"(%1) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>) -> i32
    %0 = vm.list.size %arg0 : (!vm.list<i32>) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_get_i32
  vm.func @list_get_i32(%arg0: !vm.list<i32>, %arg1: i32) -> i32 {
    // CHECK-NEXT: %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %1 = emitc.apply "&"(%0) : (!emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %2 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %3 = emitc.call_opaque "iree_vm_list_deref"(%2) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_get_value_as"(%3, %arg4, %1) {args = [0 : index, 1 : index, #emitc.opaque<"IREE_VM_VALUE_TYPE_I32">, 2 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.list.get.i32 %arg0, %arg1 : (!vm.list<i32>, i32) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_get_ref
  vm.func @list_get_ref(%arg0: !vm.list<!vm.ref<?>>, %arg1: i32) -> !vm.buffer {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call_opaque "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_get_ref_retain"(%1, %arg4, %arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    // CHECK: %[[REF_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>>
    // CHECK-NEXT: emitc.assign %arg3 : !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">> to %[[REF_LVAL]] : <!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>>
    // CHECK: %[[TYPE_LVAL:.+]] = "emitc.member_of_ptr"(%[[REF_LVAL]]) <{member = "type"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>>) -> !emitc.lvalue<!emitc.opaque<"iree_vm_ref_type_t">>
    // CHECK-NEXT: %[[TYPE:.+]] = emitc.load %[[TYPE_LVAL]] : <!emitc.opaque<"iree_vm_ref_type_t">>
    // CHECK: %[[B:.+]] = "emitc.constant"() <{value = #emitc.opaque<"IREE_VM_REF_TYPE_NULL">}> : () -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %[[C:.+]] = emitc.cmp ne, %[[TYPE]], %[[B]] : (!emitc.opaque<"iree_vm_ref_type_t">, !emitc.opaque<"iree_vm_ref_type_t">) -> i1
    // CHECK: %[[D:.+]] = emitc.call_opaque "iree_vm_type_def_is_value"(%{{.+}}) : (!emitc.opaque<"iree_vm_type_def_t">) -> i1
    // CHECK: %[[E:.+]] = emitc.call_opaque "iree_vm_type_def_as_ref"(%{{.+}}) : (!emitc.opaque<"iree_vm_type_def_t">) -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %[[F:.+]] = emitc.cmp ne, %[[TYPE]], %[[E]] : (!emitc.opaque<"iree_vm_ref_type_t">, !emitc.opaque<"iree_vm_ref_type_t">) -> i1
    // CHECK: %[[G:.+]] = emitc.logical_or %[[D]], %[[F]] : i1, i1
    // CHECK: %{{.+}} = emitc.logical_and %[[C]], %[[G]] : i1, i1
    // CHECK: cf.cond_br %{{.+}}, ^[[FAIL:.+]], ^[[CONTINUE:.+]]
    // CHECK: ^[[FAIL]]:
    // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_release"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()
    // CHECK-NEXT: cf.br ^[[CONTINUE]]
    %0 = vm.list.get.ref %arg0, %arg1 : (!vm.list<!vm.ref<?>>, i32) -> !vm.buffer
    vm.return %0 : !vm.buffer
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_set_i32
  vm.func @list_set_i32(%arg0: !vm.list<i32>, %arg1: i32, %arg2: i32) {
    // CHECK-NEXT: %0 = emitc.call_opaque "iree_vm_value_make_i32"(%arg5) : (i32) -> !emitc.opaque<"iree_vm_value_t">
    // CHECK-NEXT: %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: emitc.assign %0 : !emitc.opaque<"iree_vm_value_t"> to %1 : <!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %2 = emitc.apply "&"(%1) : (!emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %3 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %4 = emitc.call_opaque "iree_vm_list_deref"(%3) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_set_value"(%4, %arg4, %2) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.opaque<"iree_status_t">
    vm.list.set.i32 %arg0, %arg1, %arg2 : (!vm.list<i32>, i32, i32)
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_set_ref
  vm.func @list_set_ref(%arg0: !vm.list<!vm.ref<?>>, %arg1: i32, %arg2: !vm.buffer) {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call_opaque "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_set_ref_retain"(%1, %arg4, %arg5) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    vm.list.set.ref %arg0, %arg1, %arg2 : (!vm.list<!vm.ref<?>>, i32, !vm.buffer)
    vm.return
  }
}
