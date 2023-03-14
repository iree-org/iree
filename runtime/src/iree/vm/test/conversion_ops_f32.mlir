vm.module @conversion_ops_f32 {

  //===----------------------------------------------------------------------===//
  // Casting and type conversion/emulation
  //===----------------------------------------------------------------------===//

  // 5.5 f32 (0x40b00000 hex) -> 1085276160 int32
  vm.export @test_bitcast_i32_f32
  vm.func @test_bitcast_i32_f32() {
    %c1 = vm.const.i32 1085276160
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.bitcast.i32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 5.5
    vm.check.eq %v, %c2, "bitcast i32 to f32" : f32
    vm.return
  }

  // 1085276160 int32 (0x40b00000 hex) -> 5.5 f32
  vm.export @test_bitcast_f32_i32
  vm.func @test_bitcast_f32_i32() {
    %c1 = vm.const.f32 5.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.bitcast.f32.i32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 1085276160
    vm.check.eq %v, %c2, "bitcast f32 to i32" : i32
    vm.return
  }

  vm.export @test_cast_si32_f32_int_max
  vm.func @test_cast_si32_f32_int_max() {
    %c1 = vm.const.i32 2147483647
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.cast.si32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 2147483647.0
    vm.check.eq %v, %c2, "cast signed integer to a floating-point value" : f32
    vm.return
  }

  vm.export @test_cast_si32_f32_int_min
  vm.func @test_cast_si32_f32_int_min() {
    %c1 = vm.const.i32 -2147483648
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.cast.si32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 -2147483648.0
    vm.check.eq %v, %c2, "cast signed integer to a floating-point value" : f32
    vm.return
  }

  vm.export @test_cast_ui32_f32_int_max
  vm.func @test_cast_ui32_f32_int_max() {
    %c1 = vm.const.i32 4294967295
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.cast.ui32.f32 %c1dno : i32 -> f32
    %c2 = vm.const.f32 4294967295.0
    vm.check.eq %v, %c2, "cast unsigned integer to a floating-point value" : f32
    vm.return
  }

  vm.export @test_cast_f32_si32_int_max
  vm.func @test_cast_f32_si32_int_max() {
    // This is the maximum value that is representable precisely as both i32
    // and f32. An exponent of 30 with all mantissa bits set.
    %c1 = vm.const.f32 0x4effffff
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.cast.f32.si32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 0x7FFFFF80
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i32
    vm.return
  }

  vm.export @test_cast_f32_si32_int_min
  vm.func @test_cast_f32_si32_int_min() {
    %c1 = vm.const.f32 -2147483648.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.cast.f32.si32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 -2147483648
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i32
    vm.return
  }

  vm.export @test_cast_f32_si32_away_from_zero_pos
  vm.func @test_cast_f32_si32_away_from_zero_pos() {
    %c1 = vm.const.f32 2.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.cast.f32.si32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 3
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i32
    vm.return
  }

  vm.export @test_cast_f32_si32_away_from_zero_neg
  vm.func @test_cast_f32_si32_away_from_zero_neg() {
    %c1 = vm.const.f32 -2.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.cast.f32.si32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 -3
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i32
    vm.return
  }

  vm.export @test_cast_f32_ui32_int_big
  vm.func @test_cast_f32_ui32_int_big() {
    // This is the maximum value that is representable precisely as both ui32
    // and f32. An exponent of 31 with all mantissa bits set.
    %c1 = vm.const.f32 0x4f7fffff
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.cast.f32.ui32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 0xFFFFFF00
    vm.check.eq %v, %c2, "cast floating-point value to an unsigned integer" : i32
    vm.return
  }

  vm.export @test_cast_f32_ui32_away_from_zero
  vm.func @test_cast_f32_ui32_away_from_zero() {
    %c1 = vm.const.f32 2.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.cast.f32.ui32 %c1dno : f32 -> i32
    %c2 = vm.const.i32 3
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i32
    vm.return
  }

}
