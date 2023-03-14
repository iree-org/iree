vm.module @global_ops_i64 {

  //===--------------------------------------------------------------------===//
  // global.i64
  //===--------------------------------------------------------------------===//

  vm.global.i64 private @c42 = 42 : i64
  vm.global.i64 private mutable @c107_mut = 107 : i64

  vm.export @test_global_load_i64
  vm.func @test_global_load_i64() {
    %actual = vm.global.load.i64 @c42 : i64
    %expected = vm.const.i64 42
    vm.check.eq %actual, %expected, "@c42 != 42" : i64
    vm.return
  }

  vm.export @test_global_store_i64
  vm.func @test_global_store_i64() {
    %c17 = vm.const.i64 17
    vm.global.store.i64 %c17, @c107_mut : i64
    %actual = vm.global.load.i64 @c107_mut : i64
    vm.check.eq %actual, %c17, "@c107_mut != 17" : i64
    vm.return
  }

}
