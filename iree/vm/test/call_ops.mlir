vm.module @call_ops {

  vm.export @test_call_void
  vm.func @test_call_void() {
    vm.call @_v_v() : () -> ()
    vm.return
  }

  vm.export @fail_call_void
  vm.func @fail_call_void() {
    vm.call @_v_v_fail() : () -> ()
    vm.return
  }

  vm.export @test_call_i32
  vm.func @test_call_i32() {
    %c1 = vm.const.i32 1 : i32
    %0 = vm.call @_v_i() : () -> (i32)
    vm.check.eq %0, %c1, "_v_i()=1" : i32
    vm.return
  }

  vm.func @_v_v() attributes {noinline} {
    vm.return
  }

  vm.func @_v_v_fail() attributes {noinline} {
    %c2 = vm.const.i32 2 : i32
    vm.fail %c2
  }

  vm.func @_v_i() -> i32 attributes {noinline} {
    %c1 = vm.const.i32 1 : i32
    vm.return %c1 : i32
  }

}
