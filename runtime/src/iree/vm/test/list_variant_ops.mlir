vm.module @list_variant_ops {

  //===--------------------------------------------------------------------===//
  // vm.list.* with list types (nesting)
  //===--------------------------------------------------------------------===//

  vm.export @test_listception
  vm.func @test_listception() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    %c3 = vm.const.i32 3
    %c100 = vm.const.i32 100
    %c101 = vm.const.i32 101
    %c102 = vm.const.i32 102

    // [100, 101, 102]
    %inner0 = vm.list.alloc %c3 : (i32) -> !vm.list<i32>
    vm.list.resize %inner0, %c3 : (!vm.list<i32>, i32)
    vm.list.set.i32 %inner0, %c0, %c100 : (!vm.list<i32>, i32, i32)
    vm.list.set.i32 %inner0, %c1, %c101 : (!vm.list<i32>, i32, i32)
    vm.list.set.i32 %inner0, %c2, %c102 : (!vm.list<i32>, i32, i32)

    // [102, 101, 100]
    %inner1 = vm.list.alloc %c3 : (i32) -> !vm.list<i32>
    vm.list.resize %inner1, %c3 : (!vm.list<i32>, i32)
    vm.list.set.i32 %inner1, %c0, %c102 : (!vm.list<i32>, i32, i32)
    vm.list.set.i32 %inner1, %c1, %c101 : (!vm.list<i32>, i32, i32)
    vm.list.set.i32 %inner1, %c2, %c100 : (!vm.list<i32>, i32, i32)

    // [ [100, 101, 102], [102, 101, 100] ]
    %capacity = vm.const.i32 8
    %outer = vm.list.alloc %capacity : (i32) -> !vm.list<!vm.list<i32>>
    vm.list.resize %outer, %c2 : (!vm.list<!vm.list<i32>>, i32)
    vm.list.set.ref %outer, %c0, %inner0 : (!vm.list<!vm.list<i32>>, i32, !vm.list<i32>)
    vm.list.set.ref %outer, %c1, %inner1 : (!vm.list<!vm.list<i32>>, i32, !vm.list<i32>)

    %inner0_ret = vm.list.get.ref %outer, %c0 : (!vm.list<!vm.list<i32>>, i32) -> !vm.list<i32>
    vm.check.eq %inner0_ret, %inner0 : !vm.list<i32>
    %inner0_e2 = vm.list.get.i32 %inner0_ret, %c2 : (!vm.list<i32>, i32) -> i32
    vm.check.eq %inner0_e2, %c102 : i32

    %inner1_ret = vm.list.get.ref %outer, %c1 : (!vm.list<!vm.list<i32>>, i32) -> !vm.list<i32>
    vm.check.eq %inner1_ret, %inner1 : !vm.list<i32>
    %inner1_e2 = vm.list.get.i32 %inner1_ret, %c2 : (!vm.list<i32>, i32) -> i32
    vm.check.eq %inner1_e2, %c100 : i32

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with variant types
  //===--------------------------------------------------------------------===//

  vm.rodata private @byte_buffer dense<[1, 2, 3]> : tensor<3xi32>

  vm.export @test_variant
  vm.func @test_variant() {
    %capacity = vm.const.i32 42
    %list = vm.list.alloc %capacity : (i32) -> !vm.list<?>
    vm.list.resize %list, %capacity : (!vm.list<?>, i32)

    // Access element 10 as an i32.
    %c10 = vm.const.i32 10
    %v10_i32 = vm.const.i32 1234
    vm.list.set.i32 %list, %c10, %v10_i32 : (!vm.list<?>, i32, i32)
    %e10_i32 = vm.list.get.i32 %list, %c10 : (!vm.list<?>, i32) -> i32
    vm.check.eq %e10_i32, %v10_i32 : i32

    // Access element 10 as an i64.
    %v10_i64 = vm.const.i64 1234
    vm.list.set.i64 %list, %c10, %v10_i64 : (!vm.list<?>, i32, i64)
    %e10_i64 = vm.list.get.i64 %list, %c10 : (!vm.list<?>, i32) -> i64
    vm.check.eq %e10_i64, %v10_i64 : i64

    // Access element 11 as a ref object.
    %c11 = vm.const.i32 11
    %v11_buf = vm.const.ref.rodata @byte_buffer : !vm.buffer
    vm.list.set.ref %list, %c11, %v11_buf : (!vm.list<?>, i32, !vm.buffer)
    %e11_buf = vm.list.get.ref %list, %c11 : (!vm.list<?>, i32) -> !vm.buffer
    vm.check.eq %e11_buf, %v11_buf : !vm.buffer

    // Access element 11 as a different kind of ref object (incompatible).
    // Should return null.
    %e11_bad = vm.list.get.ref %list, %c11 : (!vm.list<?>, i32) -> !vm.list<i8>
    %null = vm.const.ref.zero : !vm.list<i8>
    vm.check.eq %e11_bad, %null : !vm.list<i8>

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Failure tests
  //===--------------------------------------------------------------------===//

  vm.export @fail_uninitialized_access
  vm.func @fail_uninitialized_access() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1

    %ref = vm.const.ref.rodata @byte_buffer : !vm.buffer
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<?>

    vm.list.set.ref %list, %c0, %ref : (!vm.list<?>, i32, !vm.buffer)
    vm.return
  }

  vm.export @fail_out_of_bounds_read
  vm.func @fail_out_of_bounds_read() {
    %c1 = vm.const.i32 1

    %list = vm.list.alloc %c1 : (i32) -> !vm.list<?>
    vm.list.resize %list, %c1 : (!vm.list<?>, i32)

    %ref = vm.list.get.ref %list, %c1 : (!vm.list<?>, i32) -> !vm.buffer
    %ref_dno = util.optimization_barrier %ref : !vm.buffer
    vm.return
  }

  vm.export @fail_out_of_bounds_write
  vm.func @fail_out_of_bounds_write() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1

    %ref = vm.const.ref.rodata @byte_buffer : !vm.buffer
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<?>
    vm.list.resize %list, %c1 : (!vm.list<?>, i32)

    vm.list.set.ref %list, %c1, %ref : (!vm.list<?>, i32, !vm.buffer)
    vm.return
  }

  vm.export @fail_variant_slot_change
  vm.func @fail_variant_slot_change() {
    %capacity = vm.const.i32 42
    %list = vm.list.alloc %capacity : (i32) -> !vm.list<?>
    vm.list.resize %list, %capacity : (!vm.list<?>, i32)

    %c10 = vm.const.i32 10

    // Access element 10 as an i32.
    %v10_i32 = vm.const.i32 1234
    vm.list.set.i32 %list, %c10, %v10_i32 : (!vm.list<?>, i32, i32)
    %e10_i32 = vm.list.get.i32 %list, %c10 : (!vm.list<?>, i32) -> i32
    vm.check.eq %e10_i32, %v10_i32 : i32

    // Access element 10 as a ref object.
    %v10_buf = vm.const.ref.rodata @byte_buffer : !vm.buffer
    vm.list.set.ref %list, %c10, %v10_buf : (!vm.list<?>, i32, !vm.buffer)
    %e10_buf = vm.list.get.ref %list, %c10 : (!vm.list<?>, i32) -> !vm.buffer
    vm.check.eq %e10_buf, %v10_buf : !vm.buffer

    // Accessing it as an i32 now that it stores the ref should fail at runtime.
    // TODO(benvanik): support type queries and/or make this silently return 0.
    %e10_any = vm.list.get.i32 %list, %c10 : (!vm.list<?>, i32) -> i32
    // -- FAILURE HERE --
    %zero = vm.const.i32.zero
    vm.check.eq %e10_any, %zero : i32

    vm.return
  }
}
