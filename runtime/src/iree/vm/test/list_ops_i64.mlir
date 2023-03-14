vm.module @list_ops_i64 {

  //===--------------------------------------------------------------------===//
  // vm.list.* with I64 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i64
  vm.func @test_i64() {
    %capacity = vm.const.i32 42
    %index = vm.const.i32 41
    %max_int_plus_1 = vm.const.i64 2147483648
    %list = vm.list.alloc %capacity : (i32) -> !vm.list<i64>
    %sz = vm.list.size %list : (!vm.list<i64>) -> i32
    vm.list.resize %list, %capacity : (!vm.list<i64>, i32)
    vm.list.set.i64 %list, %index, %max_int_plus_1 : (!vm.list<i64>, i32, i64)
    %v = vm.list.get.i64 %list, %index : (!vm.list<i64>, i32) -> i64
    vm.check.eq %v, %max_int_plus_1, "list<i64>.empty.set(41, MAX_INT_PLUS_1).get(41)=MAX_INT_PLUS_1" : i64
    vm.return
  }

}
