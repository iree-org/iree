vm.module @list_ops {

  //===--------------------------------------------------------------------===//
  // vm.list.* with I8 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i8
  vm.func @test_i8() {
    %c42 = vm.const.i32 42 : i32
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i8>
    %sz = vm.list.size %list : (!vm.list<i8>) -> i32
    %sz_dno = iree.do_not_optimize(%sz) : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I32 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i32
  vm.func @test_i32() {
    %c42 = vm.const.i32 42 : i32
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i32>
    %sz = vm.list.size %list : (!vm.list<i32>) -> i32
    %c100 = vm.const.i32 100 : i32
    %c101 = vm.const.i32 101 : i32
    vm.list.resize %list, %c101 : (!vm.list<i32>, i32)
    vm.list.set.i32 %list, %c100, %c42 : (!vm.list<i32>, i32, i32)
    %v = vm.list.get.i32 %list, %c100 : (!vm.list<i32>, i32) -> i32
    vm.check.eq %v, %c42 : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I64 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i64
  vm.func @test_i64() {
    %capacity = vm.const.i32 42 : i32
    %index = vm.const.i32 41 : i32
    %max_int_plus_1 = vm.const.i64 2147483648 : i64
    %list = vm.list.alloc %capacity : (i32) -> !vm.list<i64>
    %sz = vm.list.size %list : (!vm.list<i64>) -> i32
    vm.list.resize %list, %capacity : (!vm.list<i64>, i32)
    vm.list.set.i64 %list, %index, %max_int_plus_1 : (!vm.list<i64>, i32, i64)
    %v = vm.list.get.i64 %list, %index : (!vm.list<i64>, i32) -> i64
    vm.check.eq %v, %max_int_plus_1 : i64
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
}
