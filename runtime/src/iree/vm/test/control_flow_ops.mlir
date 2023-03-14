vm.module @control_flow_ops {

  //===--------------------------------------------------------------------===//
  // vm.return
  //===--------------------------------------------------------------------===//

  vm.export @test_return_empty
  vm.func @test_return_empty() {
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.fail
  //===--------------------------------------------------------------------===//

  vm.export @fail_always
  vm.func @fail_always() {
    %code = vm.const.i32 4
    vm.fail %code, "error!"
  }

  //===--------------------------------------------------------------------===//
  // vm.check.*
  //===--------------------------------------------------------------------===//

  vm.export @test_check_eq_always
  vm.func @test_check_eq_always() {
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    vm.check.eq %c1, %c1dno, "error!" : i32
    vm.return
  }

  vm.export @fail_check_eq_never
  vm.func @fail_check_eq_never() {
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    %c1dno = util.optimization_barrier %c1 : i32
    %c2dno = util.optimization_barrier %c2 : i32
    vm.check.eq %c1dno, %c2dno, "error!" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.import.resolved
  //===--------------------------------------------------------------------===//

  vm.import private optional @reserved.optional(%arg0: i32) -> i32

  // The optional import should not be found.
  vm.export @test_optional_import_resolved
  vm.func @test_optional_import_resolved() {
    %c1 = vm.const.i32 1
    %has_reserved_optional = vm.import.resolved @reserved.optional : i32
    vm.check.ne %has_reserved_optional, %c1, "missing optional import found" : i32
    vm.return
  }

  // The call should fail at runtime because the optional import is not resolved.
  vm.export @fail_optional_import_call
  vm.func @fail_optional_import_call() {
    %c1 = vm.const.i32 1
    %0 = vm.call @reserved.optional(%c1) : (i32) -> i32
    %code = vm.const.i32 4
    vm.fail %code, "unreachable!"
  }

  //===--------------------------------------------------------------------===//
  // vm.cond_br
  //===--------------------------------------------------------------------===//

  vm.export @test_cond_br
  vm.func @test_cond_br() {
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    vm.cond_br %c1dno, ^bb1, ^bb2
  ^bb1:
    vm.check.eq %c1dno, %c1dno, "error!" : i32
    vm.return
  ^bb2:
    %code = vm.const.i32 4
    vm.fail %code, "unreachable!"
  }

  vm.export @test_cond_br_int_arg
  vm.func @test_cond_br_int_arg() {
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    vm.cond_br %c1dno, ^bb1(%c1dno : i32), ^bb2(%c1dno : i32)
  ^bb1(%arg1 : i32):
    vm.check.eq %arg1, %c1dno, "error!" : i32
    vm.return
  ^bb2(%arg2 : i32):
    %code = vm.const.i32 4
    vm.fail %code, "unreachable!"
  }

  vm.export @test_cond_br_ref_arg
  vm.func @test_cond_br_ref_arg() {
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %ref = vm.const.ref.zero : !vm.ref<?>
    vm.cond_br %c1dno, ^bb1(%ref : !vm.ref<?>), ^bb2(%ref : !vm.ref<?>)
  ^bb1(%arg1 : !vm.ref<?>):
    vm.check.eq %arg1, %ref, "error!" : !vm.ref<?>
    vm.return
  ^bb2(%arg2 : !vm.ref<?>):
    %code = vm.const.i32 4
    vm.fail %code, "unreachable!"
  }

  // TODO(simon-camp): The EmitC conversion replaces vm.cond_br with cf.cond_br
  // operations. If both successor blocks are the same, these get canonicalized
  // to arith.select operations followed by an unconditional branch.
  vm.export @test_cond_br_same_successor attributes {emitc.exclude}
  vm.func private @test_cond_br_same_successor() {
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    vm.cond_br %c1dno, ^bb1(%c1dno : i32), ^bb1(%c2dno : i32)
  ^bb1(%arg1 : i32):
    vm.check.eq %arg1, %c1dno, "error!" : i32
    vm.return
  }

  vm.rodata private @buffer_a dense<[1]> : tensor<1xi8>
  vm.rodata private @buffer_b dense<[2]> : tensor<1xi8>
  vm.rodata private @buffer_c dense<[3]> : tensor<1xi8>
  vm.export @test_return_arg_cycling
  vm.func @test_return_arg_cycling() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_c = vm.const.ref.rodata @buffer_c : !vm.buffer

    %res:3 = vm.call @_return_arg_cycling(%ref_a, %ref_b, %ref_c)
        : (!vm.buffer, !vm.buffer, !vm.buffer) -> (!vm.buffer, !vm.buffer, !vm.buffer)
    vm.check.eq %res#0, %ref_b : !vm.buffer
    vm.check.eq %res#1, %ref_c : !vm.buffer
    vm.check.eq %res#2, %ref_a : !vm.buffer

    vm.return
  }

  vm.func private @_return_arg_cycling(%arg0 : !vm.buffer, %arg1: !vm.buffer,
                                       %arg2: !vm.buffer)
      -> (!vm.buffer, !vm.buffer, !vm.buffer) attributes {noinline} {
    vm.return %arg1, %arg2, %arg0 : !vm.buffer, !vm.buffer, !vm.buffer
  }

  vm.export @test_branch_arg_cycling
  vm.func @test_branch_arg_cycling() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_c = vm.const.ref.rodata @buffer_c : !vm.buffer
    %cond = vm.const.i32 0

    %res:3 = vm.call @_branch_arg_cycling(%ref_a, %ref_b, %ref_c, %cond)
        : (!vm.buffer, !vm.buffer, !vm.buffer, i32) -> (!vm.buffer, !vm.buffer, !vm.buffer)
    vm.check.eq %res#0, %ref_b : !vm.buffer
    vm.check.eq %res#1, %ref_c : !vm.buffer
    vm.check.eq %res#2, %ref_a : !vm.buffer

    vm.return
  }

  vm.func private @_branch_arg_cycling(%arg0 : !vm.buffer, %arg1: !vm.buffer,
                                       %arg2: !vm.buffer, %arg3: i32)
      -> (!vm.buffer, !vm.buffer, !vm.buffer) attributes {noinline} {
    vm.cond_br %arg3,
               ^bb1(%arg0, %arg1, %arg2: !vm.buffer, !vm.buffer, !vm.buffer),
               ^bb2(%arg1, %arg2, %arg0, %arg3: !vm.buffer, !vm.buffer, !vm.buffer, i32)
  ^bb1(%a: !vm.buffer, %b: !vm.buffer, %c: !vm.buffer):
    vm.return %a, %b, %c : !vm.buffer, !vm.buffer, !vm.buffer
  ^bb2(%d: !vm.buffer, %e: !vm.buffer, %f: !vm.buffer, %g: i32):
    vm.call @_side_effect(%g) : (i32) -> ()
    vm.return %d, %e, %f : !vm.buffer, !vm.buffer, !vm.buffer
  }

  vm.func private @_side_effect(%arg0: i32) attributes {noinline}
  {
    vm.return
  }
}
