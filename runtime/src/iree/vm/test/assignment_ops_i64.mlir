vm.module @assignment_ops_i64 {

  //===--------------------------------------------------------------------===//
  // ExtI64: Conditional assignment
  //===--------------------------------------------------------------------===//

  vm.export @test_select_i64
  vm.func @test_select_i64() {
    %c0 = vm.const.i32 0
    %c0dno = util.optimization_barrier %c0 : i32
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i64 0
    %c3 = vm.const.i64 1
    %v1 = vm.select.i64 %c0dno, %c2, %c3 : i64
    vm.check.eq %v1, %c3, "0 ? 0 : 1 = 1" : i64
    %v2 = vm.select.i64 %c1dno, %c2, %c3 : i64
    vm.check.eq %v2, %c2, "1 ? 0 : 1 = 0" : i64
    vm.return
  }
}
