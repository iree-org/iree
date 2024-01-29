vm.module @add_module {
  vm.func @add_and_double(%arg0 : i32, %arg1 : i32) -> i32 attributes {inlining_policy = #util.inline.never} {
    %0 = vm.add.i32 %arg0, %arg1 : i32
    %1 = vm.add.i32 %0, %0 : i32
    vm.return %1 : i32
  }
  vm.export @add_and_double

  vm.func @test_call(%arg0: i32) -> i32 {
    %0 = vm.call @add_and_double(%arg0, %arg0) : (i32, i32) -> i32
    %1 = vm.add.i32 %0, %0 : i32
    vm.return %1 : i32
  }
  vm.export @test_call
}
