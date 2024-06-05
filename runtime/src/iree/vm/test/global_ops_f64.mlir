vm.module @global_ops_f64 {

  //===--------------------------------------------------------------------===//
  // global.f64
  //===--------------------------------------------------------------------===//

  vm.global.f64 private @c42 = 42.5 : f64
  vm.global.f64 private mutable @c107_mut = 107.5 : f64

  vm.export @test_global_load_f64
  vm.func @test_global_load_f64() {
    %actual = vm.global.load.f64 @c42 : f64
    %expected = vm.const.f64 42.5
    vm.check.eq %actual, %expected, "@c42 != 42.5" : f64
    vm.return
  }

  vm.export @test_global_store_f64
  vm.func @test_global_store_f64() {
    %c17 = vm.const.f64 17.5
    vm.global.store.f64 %c17, @c107_mut : f64
    %actual = vm.global.load.f64 @c107_mut : f64
    vm.check.eq %actual, %c17, "@c107_mut != 17.5" : f64
    vm.return
  }

}
