vm.module @global_ops {

  //===--------------------------------------------------------------------===//
  // global.i32
  //===--------------------------------------------------------------------===//

  vm.global.i32 private @c42 = 42 : i32
  vm.global.i32 private mutable @c107_mut = 107 : i32
  vm.global.ref mutable @g0 : !vm.buffer

  vm.rodata private @buffer dense<[1, 2, 3]> : tensor<3xi8>

  vm.export @test_global_load_i32
  vm.func @test_global_load_i32() {
    %actual = vm.global.load.i32 @c42 : i32
    %expected = vm.const.i32 42
    vm.check.eq %actual, %expected, "@c42 != 42" : i32
    vm.return
  }

  vm.export @test_global_load_ref
  vm.func @test_global_load_ref() {
    %actual = vm.global.load.ref @g0 : !vm.buffer
    %expected = vm.const.ref.zero : !vm.buffer
    %expecteddno = util.optimization_barrier %expected : !vm.buffer
    vm.check.eq %actual, %expecteddno : !vm.buffer
    vm.return
  }

  vm.export @test_global_store_i32
  vm.func @test_global_store_i32() {
    %c17 = vm.const.i32 17
    vm.global.store.i32 %c17, @c107_mut : i32
    %actual = vm.global.load.i32 @c107_mut : i32
    vm.check.eq %actual, %c17, "@c107_mut != 17" : i32
    vm.return
  }

  vm.export @test_global_store_ref
  vm.func @test_global_store_ref() {
    %ref_buffer = vm.const.ref.rodata @buffer : !vm.buffer
    vm.global.store.ref %ref_buffer, @g0 : !vm.buffer
    %actual = vm.global.load.ref @g0 : !vm.buffer
    vm.check.eq %actual, %ref_buffer, "@g0 != buffer" : !vm.buffer
    vm.return
  }

}
