vm.module @module_a {
  vm.import private @module_b.square(%arg : i32) -> i32

  vm.func @test_call(%arg0: i32) -> i32 {
    %0 = vm.call @module_b.square(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
  vm.export @test_call
}
