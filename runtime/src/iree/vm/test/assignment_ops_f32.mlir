vm.module @assignment_ops_f32 {

  //===--------------------------------------------------------------------===//
  // ExtF32: Conditional assignment
  //===--------------------------------------------------------------------===//

  vm.export @test_select_f32
  vm.func @test_select_f32() {
    %c0 = vm.const.i32 0
    %c0dno = util.optimization_barrier %c0 : i32
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.f32 0.0
    %c3 = vm.const.f32 1.0
    %v1 = vm.select.f32 %c0dno, %c2, %c3 : f32
    vm.check.eq %v1, %c3, "0 ? 0.0 : 1.0 = 1.0" : f32
    %v2 = vm.select.f32 %c1dno, %c2, %c3 : f32
    vm.check.eq %v2, %c2, "1 ? 0.0 : 1.0 = 0.0" : f32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // ExtF32: Lookup table
  //===--------------------------------------------------------------------===//

  vm.export @test_switch_f32 attributes {emitc.exclude}
  vm.func private @test_switch_f32() {
    %c100 = vm.const.f32 100.0
    %c200 = vm.const.f32 200.0
    %c300 = vm.const.f32 300.0

    %i0 = vm.const.i32 0
    %i0_dno = util.optimization_barrier %i0 : i32
    %v0 = vm.switch.f32 %i0_dno[%c100, %c200] else %c300 : f32
    vm.check.eq %v0, %c100, "index 0 is 100" : f32

    %i1 = vm.const.i32 1
    %i1_dno = util.optimization_barrier %i1 : i32
    %v1 = vm.switch.f32 %i1_dno[%c100, %c200] else %c300 : f32
    vm.check.eq %v1, %c200, "index 1 is 200" : f32

    %i2 = vm.const.i32 2
    %i2_dno = util.optimization_barrier %i2 : i32
    %v2 = vm.switch.f32 %i2_dno[%c100, %c200] else %c300 : f32
    vm.check.eq %v2, %c300, "index 2 (out of bounds) is default 300" : f32

    vm.return
  }
}
