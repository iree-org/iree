// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: @my_module_buffer_alloc
vm.module @my_module {
  vm.func @buffer_alloc() {
    // CHECK: %[[SIZE:.+]] = "emitc.constant"() <{value = 128 : i64}> : () -> i64
    // CHECK-DAG: %[[ALIGNMENT:.+]] = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    // CHECK-DAG: %[[BUFFER:.+]] = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>
    // CHECK-DAG: %[[BUFFER_PTR:.+]] = emitc.apply "&"(%[[BUFFER]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>) -> !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>>
    // CHECK-DAG: %[[ALLOCTOR:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"allocator">]} : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.opaque<"iree_allocator_t">
    // CHECK-DAG: %[[BUFFER_ACCESS:.+]] = "emitc.constant"() <{value = #emitc.opaque<"IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST">}> : () -> !emitc.opaque<"iree_vm_buffer_access_t">
    // CHECK-NEXT: %[[STATUS:.+]] = emitc.call_opaque "iree_vm_buffer_create"(%[[BUFFER_ACCESS]], %[[SIZE]], %[[ALIGNMENT]], %[[ALLOCTOR]], %[[BUFFER_PTR]]) : (!emitc.opaque<"iree_vm_buffer_access_t">, i64, i32, !emitc.opaque<"iree_allocator_t">, !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>>) -> !emitc.opaque<"iree_status_t">

    // CHECK: %[[BUFFER_TYPE_ID:.+]] = emitc.call_opaque "iree_vm_buffer_type"() : () -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK-NEXT: %[[STATUS2:.+]] = emitc.call_opaque "iree_vm_ref_wrap_assign"(%[[BUFFER]], %[[BUFFER_TYPE_ID]], %1) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, !emitc.opaque<"iree_vm_ref_type_t">, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">

    %c128 = vm.const.i64 128
    %alignment = vm.const.i32 32
    %buf = vm.buffer.alloc %c128, %alignment : !vm.buffer
    vm.return
  }
}

// -----

// CHECK-LABEL: @my_module_buffer_clone
vm.module @my_module {
  vm.func @buffer_clone(%buf : !vm.buffer) {
    // CHECK-DAG: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    // CHECK-DAG: %[[C32:.+]] = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    // CHECK-DAG: %[[ALIGNMENT:.+]] = "emitc.constant"() <{value = 64 : i32}> : () -> i32

    // CHECK: %[[BUFFER:.+]] = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>
    // CHECK-DAG: %[[BUFFER_PTR:.+]] = emitc.apply "&"(%[[BUFFER]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>) -> !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>>
    // CHECK-DAG: %[[ALLOCATOR:.+]] = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"allocator">]} : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.opaque<"iree_allocator_t">
    // CHECK-DAG: %[[BUFFER_ACCESS:.+]] = "emitc.constant"() <{value = #emitc.opaque<"IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST">}> : () -> !emitc.opaque<"iree_vm_buffer_access_t">
    // CHECK-DAG: %[[BUFFER_REF2:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-DAG: %[[BUFFER_PTR2:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF2]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "iree_vm_buffer_clone"(%[[BUFFER_ACCESS]], %[[BUFFER_PTR2]], %[[C0]], %[[C32]], %[[ALIGNMENT]], %[[ALLOCATOR]], %[[BUFFER_PTR]]) : (!emitc.opaque<"iree_vm_buffer_access_t">, !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, i64, i32, !emitc.opaque<"iree_allocator_t">, !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>>) -> !emitc.opaque<"iree_status_t">
    %c0 = vm.const.i64 0
    %c32 = vm.const.i64 32
    %alignment = vm.const.i32 64
    %buf_clone = vm.buffer.clone %buf, %c0, %c32, %alignment : !vm.buffer -> !vm.buffer
    vm.return
  }
}

// -----

// CHECK-LABEL: @my_module_buffer_length
vm.module @my_module {
  vm.func @buffer_length(%buf : !vm.buffer) {
    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[LENGTH:.+]] = emitc.call_opaque "iree_vm_buffer_length"(%[[BUFFER_PTR]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>) -> i64

    %length = vm.buffer.length %buf : !vm.buffer -> i64
    vm.return
  }
}

// -----

// CHECK-LABEL: @my_module_buffer_compare
vm.module @my_module {
  vm.func @buffer_compare(%buf : !vm.buffer, %buf2 : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    // CHECK: %[[C16:.+]] = "emitc.constant"() <{value = 16 : i64}> : () -> i64

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[BUFFER_REF2:.+]] = emitc.apply "*"(%arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR2:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF2]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[RESULT:.+]] = "emitc.variable"() <{value = 0 : i32}> : () -> i32
    // CHECK-NEXT: %[[RESULT_PTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i32) -> !emitc.ptr<i32>
    // CHECK-NEXT: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_compare"(%[[BUFFER_PTR]], %[[C0]], %[[BUFFER_PTR2]], %[[C16]], %[[C16]], %[[RESULT_PTR]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, i64, !emitc.ptr<i32>) -> !emitc.opaque<"iree_status_t">
    %c0 = vm.const.i64 0
    %c16 = vm.const.i64 16
    %cmp = vm.buffer.compare %buf, %c0, %buf2, %c16, %c16 : !vm.buffer, !vm.buffer
    vm.return
  }
}

// -----

// CHECK-LABEL: @my_module_buffer_copy
vm.module @my_module {
  vm.func @buffer_copy(%buf : !vm.buffer, %buf2 : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    // CHECK: %[[C16:.+]] = "emitc.constant"() <{value = 16 : i64}> : () -> i64

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[BUFFER_REF2:.+]] = emitc.apply "*"(%arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR2:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF2]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "iree_vm_buffer_copy_bytes"(%[[BUFFER_PTR]], %[[C0]], %[[BUFFER_PTR2]], %[[C16]], %[[C16]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, i64) -> !emitc.opaque<"iree_status_t">
    %c0 = vm.const.i64 0
    %c16 = vm.const.i64 16
    vm.buffer.copy %buf, %c0, %buf2, %c16, %c16 : !vm.buffer -> !vm.buffer
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_buffer_fill_i8
  vm.func @buffer_fill_i8(%buf : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    // CHECK: %[[C16:.+]] = "emitc.constant"() <{value = 16 : i64}> : () -> i64
    // CHECK: %[[C102:.+]] = "emitc.constant"() <{value = 102 : i32}> : () -> i32

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_fill_i8"(%[[BUFFER_PTR]], %[[C0]], %[[C16]], %[[C102]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, i64, i32) -> !emitc.opaque<"iree_status_t">
    %c0 = vm.const.i64 0
    %c16 = vm.const.i64 16
    %c102 = vm.const.i32 102
    vm.buffer.fill.i8 %buf, %c0, %c16, %c102 : i32 -> !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @my_module_buffer_fill_i16
  vm.func @buffer_fill_i16(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_fill_i16"
    %c0 = vm.const.i64 0
    %c16 = vm.const.i64 16
    %c102 = vm.const.i32 102
    vm.buffer.fill.i16 %buf, %c0, %c16, %c102 : i32 -> !vm.buffer
    vm.return
  }

    // CHECK-LABEL: @my_module_buffer_fill_i32
  vm.func @buffer_fill_i32(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_fill_i32"
    %c0 = vm.const.i64 0
    %c16 = vm.const.i64 16
    %c102 = vm.const.i32 102
    vm.buffer.fill.i32 %buf, %c0, %c16, %c102 : i32 -> !vm.buffer
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_buffer_load_i8s
  vm.func @buffer_load_i8s(%buf : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[RESULT:.+]] = "emitc.variable"() <{value = 0 : i32}> : () -> i32
    // CHECK-NEXT: %[[RESULT_PTR:.+]] = emitc.apply "&"(%6) : (i32) -> !emitc.ptr<i32>
    // CHECK-NEXT: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_load_i8s"(%[[BUFFER_PTR]], %[[C0]], %[[RESULT_PTR]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, !emitc.ptr<i32>) -> !emitc.opaque<"iree_status_t">

    %c0 = vm.const.i64 0
    %v0 = vm.buffer.load.i8.s %buf[%c0] : !vm.buffer -> i32
    vm.return
  }

  // CHECK-LABEL: @my_module_buffer_load_i8u
  vm.func @buffer_load_i8u(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_load_i8u"
    %c0 = vm.const.i64 0
    %v0 = vm.buffer.load.i8.u %buf[%c0] : !vm.buffer -> i32
    vm.return
  }

  // CHECK-LABEL: @my_module_buffer_load_i16s
  vm.func @buffer_load_i16s(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_load_i16s"
    %c0 = vm.const.i64 0
    %v0 = vm.buffer.load.i16.s %buf[%c0] : !vm.buffer -> i32
    vm.return
  }

  // CHECK-LABEL: @my_module_buffer_load_i16u
  vm.func @buffer_load_i16u(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_load_i16u"
    %c0 = vm.const.i64 0
    %v0 = vm.buffer.load.i16.u %buf[%c0] : !vm.buffer -> i32
    vm.return
  }

  // CHECK-LABEL: @my_module_buffer_load_i32
  vm.func @buffer_load_i32(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_load_i32"
    %c0 = vm.const.i64 0
    %v0 = vm.buffer.load.i32 %buf[%c0] : !vm.buffer -> i32
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_buffer_store_i8
  vm.func @buffer_store_i8(%buf : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    // CHECK: %[[C102:.+]] = "emitc.constant"() <{value = 102 : i32}> : () -> i32

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_store_i8"(%[[BUFFER_PTR]], %[[C0]], %[[C102]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, i32) -> !emitc.opaque<"iree_status_t">
    %c0 = vm.const.i64 0
    %c102 = vm.const.i32 102
    vm.buffer.store.i8 %c102, %buf[%c0] : i32 -> !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @my_module_buffer_store_i16
  vm.func @buffer_store_i16(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_store_i16"
    %c0 = vm.const.i64 0
    %c102 = vm.const.i32 102
    vm.buffer.store.i16 %c102, %buf[%c0] : i32 -> !vm.buffer
    vm.return
  }

    // CHECK-LABEL: @my_module_buffer_store_i32
  vm.func @buffer_store_i32(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_store_i32"
    %c0 = vm.const.i64 0
    %c102 = vm.const.i32 102
    vm.buffer.store.i32 %c102, %buf[%c0] : i32 -> !vm.buffer
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_buffer_hash
  vm.func @buffer_hash(%buf : !vm.buffer) {
    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "iree_vm_buffer_hash"
    %c0 = vm.const.i64 0
    %c10 = vm.const.i64 10
    %v0 = vm.buffer.hash %buf, %c0, %c10 : !vm.buffer -> i64
    vm.return
  }
}
