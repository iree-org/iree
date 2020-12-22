vm.module @add_module {
  vm.func @add(%arg0 : i32, %arg1 : i32) -> i32 {
    %0 = vm.add.i32 %arg0, %arg1 : i32
    %1 = vm.add.i32 %0, %0 : i32
    vm.return %1 : i32
  }
  vm.export @add

  vm.func @add_call(%arg0: i32) -> i32 {
    %0 = vm.call @add(%arg0, %arg0) : (i32, i32) -> i32
    // TODO(simon-camp) use %0 as arguments when the call operation is properly implemented
    %1 = vm.add.i32 %arg0, %arg0 : i32
    vm.return %1 : i32
  }
  vm.export @add_call
}
