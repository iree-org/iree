vm.module @conversion_ops {

  //===----------------------------------------------------------------------===//
  // Casting and type conversion/emulation
  //===----------------------------------------------------------------------===//

  vm.export @test_trunc_i32_i8
  vm.func @test_trunc_i32_i8() {
    %c1 = vm.const.i32 2147483647 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    %v = vm.trunc.i32.i8 %c1dno : i32 -> i32
    %c2 = vm.const.i32 255 : i32
    vm.check.eq %v, %c2, "truncate unsigned i32 to unsigned i8" : i32
    vm.return
  }

  vm.export @test_trunc_i32_i16
  vm.func @test_trunc_i32_i16() {
    %c1 = vm.const.i32 2147483647 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    %v = vm.trunc.i32.i16 %c1dno : i32 -> i32
    %c2 = vm.const.i32 65535 : i32
    vm.check.eq %v, %c2, "truncate unsigned i32 to unsigned i16" : i32
    vm.return
  }

  // 5.5 f32 (0x40b00000 hex) -> 1085276160 int32
  vm.export @test_bitcast_i32_f32
  vm.func @test_bitcast_i32_f32() {
    %c1 = vm.const.i32 1085276160 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    %v = vm.bitcast.i32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 5.5 : f32
    vm.check.eq %v, %c2, "bitcast i32 to f32" : f32
    vm.return
  }

  // 1085276160 int32 (0x40b00000 hex) -> 5.5 f32
  vm.export @test_bitcast_f32_i32
  vm.func @test_bitcast_f32_i32() {
    %c1 = vm.const.f32 5.5 : f32
    %c1dno = util.do_not_optimize(%c1) : f32
    %v = vm.bitcast.f32.i32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 1085276160 : i32
    vm.check.eq %v, %c2, "bitcast f32 to i32" : i32
    vm.return
  }

}
