vm.module @add_module {
  vm.func @add_1(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    %0 = vm.add.i32 %arg0, %arg1 : i32
    %1 = vm.add.i32 %0, %0 : i32
    vm.return %0, %1 : i32, i32
  }
}
