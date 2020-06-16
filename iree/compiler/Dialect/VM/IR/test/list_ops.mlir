// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// Common operations that don't care what the type of the list is.
vm.module @module {
  // CHECK-LABEL: @list_common
  vm.func @list_common() {
    %c42 = vm.const.i32 42 : i32
    // CHECK: %list = vm.list.alloc %c42 : (i32) -> !vm.list<i32>
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i32>

    %c43 = vm.const.i32 43 : i32
    // CHECK: vm.list.reserve %list, %c43 : (!vm.list<i32>, i32)
    vm.list.reserve %list, %c43 : (!vm.list<i32>, i32)

    // CHECK: = vm.list.size %list : (!vm.list<i32>) -> i32
    %0 = vm.list.size %list : (!vm.list<i32>) -> i32

    // CHECK: vm.list.resize %list, %c44 : (!vm.list<i32>, i32)
    %c44 = vm.const.i32 44 : i32
    vm.list.resize %list, %c44 : (!vm.list<i32>, i32)

    vm.return
  }
}

// -----

// Typed accessors for lists with i32 elements.
vm.module @module {
  // CHECK: @list_i32
  vm.func @list_i32(%arg0 : !vm.list<i32>) {
    %c100 = vm.const.i32 100 : i32

    // CHECK: vm.list.get.i32 %arg0, %c100 : (!vm.list<i32>, i32) -> i32
    %0 = vm.list.get.i32 %arg0, %c100 : (!vm.list<i32>, i32) -> i32

    // CHECK: vm.list.set.i32 %arg0, %c100, %c123 : (!vm.list<i32>, i32, i32)
    %c123 = vm.const.i32 123 : i32
    vm.list.set.i32 %arg0, %c100, %c123 : (!vm.list<i32>, i32, i32)

    vm.return
  }

  // CHECK: @list_i8_coerce
  vm.func @list_i8_coerce(%arg0 : !vm.list<i8>) {
    %c100 = vm.const.i32 100 : i32

    // CHECK: = vm.list.get.i32 %arg0, %c100 : (!vm.list<i8>, i32) -> i32
    %0 = vm.list.get.i32 %arg0, %c100 : (!vm.list<i8>, i32) -> i32

    // CHECK: vm.list.set.i32 %arg0, %c100, %0 : (!vm.list<i8>, i32, i32)
    vm.list.set.i32 %arg0, %c100, %0 : (!vm.list<i8>, i32, i32)

    vm.return
  }
}

// -----

// Typed accessors for lists with opaque ref<?> elements.
vm.module @module {
  // CHECK: @list_ref_any
  vm.func @list_ref_any(%arg0 : !vm.list<!vm.ref<?>>) {
    %c100 = vm.const.i32 100 : i32

    // CHECK: %ref = vm.list.get.ref %arg0, %c100 : (!vm.list<!vm.ref<?>>, i32) -> !vm.ref<!iree.byte_buffer>
    %ref = vm.list.get.ref %arg0, %c100 : (!vm.list<!vm.ref<?>>, i32) -> !vm.ref<!iree.byte_buffer>

    // CHECK: vm.list.set.ref %arg0, %c100, %ref : (!vm.list<!vm.ref<?>>, i32, !vm.ref<!iree.byte_buffer>)
    vm.list.set.ref %arg0, %c100, %ref : (!vm.list<!vm.ref<?>>, i32, !vm.ref<!iree.byte_buffer>)

    vm.return
  }
}

// -----

// Typed accessors for lists with strongly-typed ref elements.
vm.module @module {
  // CHECK: @list_ref_typed
  vm.func @list_ref_typed(%arg0 : !vm.list<!vm.ref<!iree.byte_buffer>>) {
    %c100 = vm.const.i32 100 : i32

    // CHECK: %ref = vm.list.get.ref %arg0, %c100 : (!vm.list<!vm.ref<!iree.byte_buffer>>, i32) -> !vm.ref<!iree.byte_buffer>
    %ref = vm.list.get.ref %arg0, %c100 : (!vm.list<!vm.ref<!iree.byte_buffer>>, i32) -> !vm.ref<!iree.byte_buffer>

    // CHECK: vm.list.set.ref %arg0, %c100, %ref : (!vm.list<!vm.ref<!iree.byte_buffer>>, i32, !vm.ref<!iree.byte_buffer>)
    vm.list.set.ref %arg0, %c100, %ref : (!vm.list<!vm.ref<!iree.byte_buffer>>, i32, !vm.ref<!iree.byte_buffer>)

    vm.return
  }
}
