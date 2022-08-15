vm.module @global_ops_f32 {

  //===--------------------------------------------------------------------===//
  // global.f32
  //===--------------------------------------------------------------------===//

  vm.global.f32 private @c42 = 42.5 : f32
  vm.global.f32 private mutable @c107_mut = 107.5 : f32

  vm.export @test_global_load_f32
  vm.func @test_global_load_f32() {
    %actual = vm.global.load.f32 @c42 : f32
    %expected = vm.const.f32 42.5
    vm.check.eq %actual, %expected, "@c42 != 42.5" : f32
    vm.return
  }

  vm.export @test_global_store_f32
  vm.func @test_global_store_f32() {
    %c17 = vm.const.f32 17.5
    vm.global.store.f32 %c17, @c107_mut : f32
    %actual = vm.global.load.f32 @c107_mut : f32
    vm.check.eq %actual, %c17, "@c107_mut != 17.5" : f32
    vm.return
  }

}
