vm.module @call_ops {

  vm.export @fail_call_v_v
  vm.func @fail_call_v_v() {
    vm.call @_v_v_fail() : () -> ()
    vm.return
  }

  vm.export @test_call_i_v
  vm.func @test_call_i_v() {
    %c1 = vm.const.i32 1 : i32
    vm.call @_i_v(%c1) : (i32) -> ()
    vm.return
  }

  // TODO(simon-camp): The EmitC conversion doesn't support ref types on function boundaries.
  vm.export @test_call_r_v attributes {emitc.exclude}
  vm.func private @test_call_r_v() {
    %ref = vm.const.ref.zero : !vm.ref<?>
    vm.call @_r_v(%ref) : (!vm.ref<?>) -> ()
    vm.return
  }

  vm.export @test_call_v_i
  vm.func @test_call_v_i() {
    %c1 = vm.const.i32 1 : i32
    %0 = vm.call @_v_i() : () -> (i32)
    vm.check.eq %0, %c1, "_v_i()=1" : i32
    vm.return
  }

  // TODO(simon-camp): The EmitC conversion doesn't support ref types on function boundaries.
  vm.export @test_call_v_r attributes {emitc.exclude}
  vm.func private @test_call_v_r() {
    %ref = vm.const.ref.zero : !vm.ref<?>
    %res = vm.call @_v_r() : () -> (!vm.ref<?>)
    vm.check.eq %ref, %res, "_v_r()=NULL" : !vm.ref<?>
    vm.return
  }

  // TODO(simon-camp): The EmitC conversion doesn't support multiple return values.
  vm.export @test_call_v_ii attributes {emitc.exclude}
  vm.func private @test_call_v_ii() {
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %0:2 = vm.call @_v_ii() : () -> (i32, i32)
    vm.check.eq %0#0, %c1, "_v_ii()#0=1" : i32
    vm.check.eq %0#1, %c2, "_v_ii()#1=2" : i32
    vm.return
  }

  vm.export @test_call_v_v
  vm.func @test_call_v_v() {
    vm.call @_v_v() : () -> ()
    vm.return
  }

  vm.func @_i_v(%arg : i32) attributes {noinline} {
    vm.return
  }

  vm.func private @_r_v(%arg : !vm.ref<?>) attributes {noinline} {
    vm.return
  }

  vm.func @_v_i() -> i32 attributes {noinline} {
    %c1 = vm.const.i32 1 : i32
    vm.return %c1 : i32
  }

  vm.func private @_v_r() -> !vm.ref<?> attributes {noinline} {
    %ref = vm.const.ref.zero : !vm.ref<?>
    vm.return %ref : !vm.ref<?>
  }

  vm.func @_v_ii() -> (i32, i32) attributes {noinline} {
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    vm.return %c1, %c2 : i32, i32
  }

  vm.func @_v_v() attributes {noinline} {
    vm.return
  }

  vm.func @_v_v_fail() attributes {noinline} {
    %c2 = vm.const.i32 2 : i32
    vm.fail %c2
  }

}
