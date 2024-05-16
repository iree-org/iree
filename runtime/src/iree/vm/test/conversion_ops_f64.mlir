vm.module @conversion_ops_f64 {

  //===----------------------------------------------------------------------===//
  // Casting and type conversion/emulation
  //===----------------------------------------------------------------------===//

  vm.export @test_bitcast_i64_f64
  vm.func @test_bitcast_i64_f64() {
    %c1 = vm.const.i64 0x4016000000000000
    %c1dno = util.optimization_barrier %c1 : i64
    %v = vm.bitcast.i64.f64 %c1dno : i64 -> f64
    %c2 = vm.const.f64 5.5
    vm.check.eq %v, %c2, "bitcast i64 to f64" : f64
    vm.return
  }

  vm.export @test_bitcast_f64_i64
  vm.func @test_bitcast_f64_i64() {
    %c1 = vm.const.f64 5.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.bitcast.f64.i64 %c1dno : f64 -> i64
    %c2 = vm.const.i64 0x4016000000000000
    vm.check.eq %v, %c2, "bitcast f64 to i64" : i64
    vm.return
  }

  vm.export @test_cast_si64_f64_int_max
  vm.func @test_cast_si64_f64_int_max() {
    %c1 = vm.const.i64 2147483647
    %c1dno = util.optimization_barrier %c1 : i64
    %v = vm.cast.si64.f64 %c1dno : i64 -> f64
    %c2 = vm.const.f64 2147483647.0
    vm.check.eq %v, %c2, "cast signed integer to a floating-point value" : f64
    vm.return
  }

  vm.export @test_cast_si64_f64_int_min
  vm.func @test_cast_si64_f64_int_min() {
    %c1 = vm.const.i64 -2147483648
    %c1dno = util.optimization_barrier %c1 : i64
    %v = vm.cast.si64.f64 %c1dno : i64 -> f64
    %c2 = vm.const.f64 -2147483648.0
    vm.check.eq %v, %c2, "cast signed integer to a floating-point value" : f64
    vm.return
  }

  vm.export @test_cast_ui64_f64_int_max
  vm.func @test_cast_ui64_f64_int_max() {
    %c1 = vm.const.i64 4294967295
    %c1dno = util.optimization_barrier %c1 : i64
    %v = vm.cast.ui64.f64 %c1dno : i64 -> f64
    %c2 = vm.const.f64 4294967295.0
    vm.check.eq %v, %c2, "cast unsigned integer to a floating-point value" : f64
    vm.return
  }

  vm.export @test_cast_f64_si64_int_min
  vm.func @test_cast_f64_si64_int_min() {
    %c1 = vm.const.f64 -2147483648.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.cast.f64.si64 %c1dno : f64 -> i64
    %c2 = vm.const.i64 -2147483648
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i64
    vm.return
  }

  vm.export @test_cast_f64_si64_away_from_zero_pos
  vm.func @test_cast_f64_si64_away_from_zero_pos() {
    %c1 = vm.const.f64 2.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.cast.f64.si64 %c1dno : f64 -> i64
    %c2 = vm.const.i64 3
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i64
    vm.return
  }

  vm.export @test_cast_f64_si64_away_from_zero_neg
  vm.func @test_cast_f64_si64_away_from_zero_neg() {
    %c1 = vm.const.f64 -2.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.cast.f64.si64 %c1dno : f64 -> i64
    %c2 = vm.const.i64 -3
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i64
    vm.return
  }

  vm.export @test_cast_f64_ui64_away_from_zero
  vm.func @test_cast_f64_ui64_away_from_zero() {
    %c1 = vm.const.f64 2.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.cast.f64.ui64 %c1dno : f64 -> i64
    %c2 = vm.const.i64 3
    vm.check.eq %v, %c2, "cast floating-point value to a signed integer" : i64
    vm.return
  }

}
