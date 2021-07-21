vm.module @assignment_ops_f32 {

  //===--------------------------------------------------------------------===//
  // ExtF32: Conditional assignment
  //===--------------------------------------------------------------------===//

  vm.export @test_select_f32
  vm.func @test_select_f32() {
    %c0 = vm.const.i32 0 : i32
    %c0dno = iree.do_not_optimize(%c0) : i32
    %c1 = vm.const.i32 1 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.f32 0.0 : f32
    %c3 = vm.const.f32 1.0 : f32
    %v1 = vm.select.f32 %c0dno, %c2, %c3 : f32
    vm.check.eq %v1, %c3, "0 ? 0.0 : 1.0 = 1.0" : f32
    %v2 = vm.select.f32 %c1dno, %c2, %c3 : f32
    vm.check.eq %v2, %c2, "1 ? 0.0 : 1.0 = 0.0" : f32
    vm.return
  }
}
