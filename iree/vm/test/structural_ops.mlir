  vm.module @structural_ops {

  //===--------------------------------------------------------------------===//
  // vm.export
  //===--------------------------------------------------------------------===//
    
  vm.export @test_export_with_different_name as("test_export_alias")
  vm.func @test_export_with_different_name() {
      %c1 = vm.const.i32 1 : i32
      %c1dno = iree.do_not_optimize(%c1) : i32
      vm.check.eq %c1dno, %c1dno, "1=1" : i32
      vm.return
  }

  vm.export @test_export$special.chars
  vm.func @test_export$special.chars() {
      %c1 = vm.const.i32 1 : i32
      %c1dno = iree.do_not_optimize(%c1) : i32
      vm.check.eq %c1dno, %c1dno, "1=1" : i32
      vm.return
  }

  // EmitC specific tests

  // Test for clashes after name mangling 
  vm.export @test_func_A
  vm.func @test_func_A() {
      %c1 = vm.const.i32 1 : i32
      %c1dno = iree.do_not_optimize(%c1) : i32
      vm.check.eq %c1dno, %c1dno, "1=1" : i32
      vm.return
  }

  vm.export @test_func.A
  vm.func @test_func.A() {
      %c1 = vm.const.i32 1 : i32
      %c1dno = iree.do_not_optimize(%c1) : i32
      vm.check.eq %c1dno, %c1dno, "1=1" : i32
      vm.return
  }
}
