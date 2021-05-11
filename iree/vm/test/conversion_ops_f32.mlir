vm.module @conversion_ops {

  //===----------------------------------------------------------------------===//
  // Casting and type conversion/emulation
  //===----------------------------------------------------------------------===//

  vm.export @test_cast_si32_f32_int_max
  vm.func @test_cast_si32_f32_int_max() {
    %c1 = vm.const.i32 2147483647 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.cast.si32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 2147483647.0 : f32
    vm.check.eq %v, %c2, "cast signed integer to a floating-point value" : f32
    vm.return
  }

  vm.export @test_cast_si32_f32_int_min
  vm.func @test_cast_si32_f32_int_min() {
    %c1 = vm.const.i32 -2147483648 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.cast.si32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 -2147483648.0 : f32
    vm.check.eq %v, %c2, "cast signed integer to a floating-point value" : f32
    vm.return
  }

  vm.export @test_cast_ui32_f32_int_max
  vm.func @test_cast_ui32_f32_int_max() {
    %c1 = vm.const.i32 4294967295 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.cast.ui32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 4294967295.0 : f32
    vm.check.eq %v, %c2, "cast unsigned integer to a floating-point value" : f32
    vm.return
  }

  vm.export @test_cast_f32_si32_int_max
  vm.func @test_cast_f32_si32_int_max() {
    %c1 = vm.const.f32 2147483647.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.cast.f32.si32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 -2147483648 : i32
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i32
    vm.return
  }

  vm.export @test_cast_f32_si32_int_min
  vm.func @test_cast_f32_si32_int_min() {
    %c1 = vm.const.f32 -2147483648.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.cast.f32.si32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 -2147483648 : i32
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i32
    vm.return
  }

  vm.export @test_cast_f32_ui32_int_max
  vm.func @test_cast_f32_ui32_int_max() {
    %c1 = vm.const.f32 4294967295.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.cast.f32.ui32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 0 : i32
    vm.check.eq %v, %c2, "cast floating-point value to an unsigned integer" : i32
    vm.return
  }

}
