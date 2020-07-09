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
  // vm.list.* with ref types
  //===--------------------------------------------------------------------===//

  vm.export @test_ref
  vm.func @test_ref() {
    // TODO(benvanik): test vm.list with ref types.
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with variant types
  //===--------------------------------------------------------------------===//

  vm.export @test_variant
  vm.func @test_variant() {
    // TODO(benvanik): test vm.list with variant types.
    vm.return
  }

}
