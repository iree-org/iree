vm.module @module_b {
  vm.func @square(%arg : i32) -> i32 {
    %0 = vm.mul.i32 %arg, %arg : i32
    vm.return %0 : i32
  }
  vm.export @square
}
