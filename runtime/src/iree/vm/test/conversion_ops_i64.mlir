vm.module @conversion_ops_i64 {

  //===----------------------------------------------------------------------===//
  // ExtI64: Casting and type conversion/emulation
  //===----------------------------------------------------------------------===//

  vm.export @test_trunc_i64_i32
  vm.func @test_trunc_i64_i32() {
    %c1 = vm.const.i64 9223372036854775807
    %c1dno = util.optimization_barrier %c1 : i64
    %v = vm.trunc.i64.i32 %c1dno : i64 -> i32
    %c2 = vm.const.i32 4294967295
    vm.check.eq %v, %c2, "truncate unsigned i64 to unsigned i32" : i32
    vm.return
  }

}
