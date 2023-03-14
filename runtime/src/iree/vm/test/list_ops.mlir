vm.module @list_ops {

  //===--------------------------------------------------------------------===//
  // vm.list.* with I8 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i8
  vm.func @test_i8() {
    %c42 = vm.const.i32 42
    %c100 = vm.const.i32 100
    %c0 = vm.const.i32 0
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i8>
    vm.list.reserve %list, %c100 : (!vm.list<i8>, i32)
    %sz = vm.list.size %list : (!vm.list<i8>) -> i32
    %sz_dno = util.optimization_barrier %sz : i32
    vm.check.eq %sz_dno, %c0, "list<i8>.empty.size()=0" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I16 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i16
  vm.func @test_i16() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c27 = vm.const.i32 27
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i16>
    vm.list.resize %list, %c1 : (!vm.list<i16>, i32)
    vm.list.set.i32 %list, %c0, %c27 : (!vm.list<i16>, i32, i32)
    %v = vm.list.get.i32 %list, %c0 : (!vm.list<i16>, i32) -> i32
    vm.check.eq %v, %c27, "list<i16>.empty.set(0, 27).get(0)=27" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I32 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i32
  vm.func @test_i32() {
    %c42 = vm.const.i32 42
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i32>
    %sz = vm.list.size %list : (!vm.list<i32>) -> i32
    %c100 = vm.const.i32 100
    %c101 = vm.const.i32 101
    vm.list.resize %list, %c101 : (!vm.list<i32>, i32)
    vm.list.set.i32 %list, %c100, %c42 : (!vm.list<i32>, i32, i32)
    %v = vm.list.get.i32 %list, %c100 : (!vm.list<i32>, i32) -> i32
    vm.check.eq %v, %c42, "list<i32>.empty.set(100, 42).get(100)=42" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with ref types
  //===--------------------------------------------------------------------===//

  vm.export @test_ref
  vm.func @test_ref() {
    // TODO(benvanik): test vm.list with ref types.
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Multiple lists within the same block
  //===--------------------------------------------------------------------===//

  vm.export @test_multiple_lists
  vm.func @test_multiple_lists() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c27 = vm.const.i32 27
    %c42 = vm.const.i32 42

    // These allocs shouldn't be CSE'd.
    %list0 = vm.list.alloc %c1 : (i32) -> !vm.list<i8>
    %list1 = vm.list.alloc %c1 : (i32) -> !vm.list<i8>
    vm.list.resize %list0, %c1 : (!vm.list<i8>, i32)
    vm.list.resize %list1, %c1 : (!vm.list<i8>, i32)
    vm.list.set.i32 %list0, %c0, %c27 : (!vm.list<i8>, i32, i32)
    vm.list.set.i32 %list1, %c0, %c42 : (!vm.list<i8>, i32, i32)
    %res0 = vm.list.get.i32 %list0, %c0 : (!vm.list<i8>, i32) -> i32
    %res1 = vm.list.get.i32 %list1, %c0 : (!vm.list<i8>, i32) -> i32
    vm.check.eq %res0, %c27, "list0.get(0)=27" : i32
    vm.check.eq %res1, %c42, "list1.get(0)=42" : i32

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Failure tests
  //===--------------------------------------------------------------------===//

  vm.export @fail_uninitialized_access
  vm.func @fail_uninitialized_access() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i32>
    vm.list.set.i32 %list, %c0, %c1 : (!vm.list<i32>, i32, i32)
    vm.return
  }

  vm.export @fail_out_of_bounds_read
  vm.func @fail_out_of_bounds_read() {
    %c1 = vm.const.i32 1
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i32>
    vm.list.resize %list, %c1 : (!vm.list<i32>, i32)
    %v = vm.list.get.i32 %list, %c1 : (!vm.list<i32>, i32) -> i32
    %v_dno = util.optimization_barrier %v : i32
    // Add a dummy use of %v_dno to please recent versions of clang for the C target
    vm.list.set.i32 %list, %c1, %v_dno : (!vm.list<i32>, i32, i32)
    vm.return
  }

  vm.export @fail_out_of_bounds_write
  vm.func @fail_out_of_bounds_write() {
    %c1 = vm.const.i32 1
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i32>
    vm.list.resize %list, %c1 : (!vm.list<i32>, i32)
    vm.list.set.i32 %list, %c1, %c1 : (!vm.list<i32>, i32, i32)
    vm.return
  }
}
