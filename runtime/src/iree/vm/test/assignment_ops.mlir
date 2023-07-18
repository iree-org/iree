vm.module @assignment_ops {

  //===--------------------------------------------------------------------===//
  // Conditional assignment
  //===--------------------------------------------------------------------===//

  vm.export @test_select_i32
  vm.func @test_select_i32() {
    %c0 = vm.const.i32 0
    %c0dno = util.optimization_barrier %c0 : i32
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %v1 = vm.select.i32 %c0dno, %c0dno, %c1dno : i32
    vm.check.eq %v1, %c1, "0 ? 0 : 1 = 1" : i32
    %v2 = vm.select.i32 %c1dno, %c0dno, %c1dno : i32
    vm.check.eq %v2, %c0, "1 ? 0 : 1 = 0" : i32
    vm.return
  }

  vm.export @test_select_ref attributes {emitc.exclude}
  vm.func private @test_select_ref() {
    %c0 = vm.const.i32 0
    %list0 = vm.list.alloc %c0 : (i32) -> !vm.list<i8>
    %c1 = vm.const.i32 1
    %list1 = vm.list.alloc %c1 : (i32) -> !vm.list<i8>
    %cond = vm.const.i32 0
    %cond_dno = util.optimization_barrier %cond : i32
    %list = vm.select.ref %cond_dno, %list0, %list1 : !vm.list<i8>
    vm.check.eq %list, %list1, "0 ? list0 : list1 = list1" : !vm.list<i8>
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Lookup table
  //===--------------------------------------------------------------------===//

  vm.export @test_switch_i32 attributes {emitc.exclude}
  vm.func private @test_switch_i32() {
    %c100 = vm.const.i32 100
    %c200 = vm.const.i32 200
    %c300 = vm.const.i32 300

    %i0 = vm.const.i32 0
    %i0_dno = util.optimization_barrier %i0 : i32
    %v0 = vm.switch.i32 %i0_dno[%c100, %c200] else %c300 : i32
    vm.check.eq %v0, %c100, "index 0 is 100" : i32

    %i1 = vm.const.i32 1
    %i1_dno = util.optimization_barrier %i1 : i32
    %v1 = vm.switch.i32 %i1_dno[%c100, %c200] else %c300 : i32
    vm.check.eq %v1, %c200, "index 1 is 200" : i32

    %i2 = vm.const.i32 2
    %i2_dno = util.optimization_barrier %i2 : i32
    %v2 = vm.switch.i32 %i2_dno[%c100, %c200] else %c300 : i32
    vm.check.eq %v2, %c300, "index 2 (out of bounds) is default 300" : i32

    vm.return
  }
}
