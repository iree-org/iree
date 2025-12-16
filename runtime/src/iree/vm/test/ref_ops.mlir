vm.module @ref_ops {

  //===--------------------------------------------------------------------===//
  // Test data
  //===--------------------------------------------------------------------===//

  vm.rodata private @buffer_a dense<[1, 2, 3]> : tensor<3xi8>
  vm.rodata private @buffer_b dense<[4, 5, 6]> : tensor<3xi8>
  vm.rodata private @buffer_c dense<[7, 8, 9]> : tensor<3xi8>

  vm.global.ref private mutable @global_ref : !vm.buffer

  //===--------------------------------------------------------------------===//
  // Basic ref comparisons
  //===--------------------------------------------------------------------===//

  vm.export @test_zero_ref_eq
  vm.func @test_zero_ref_eq() {
    %ref = vm.const.ref.zero : !vm.ref<?>
    %ref_dno = vm.optimization_barrier %ref : !vm.ref<?>
    vm.check.eq %ref_dno, %ref_dno : !vm.ref<?>
    vm.return
  }

  // TODO(simon-camp): In the C target we run the DropCompilerHintsPass after
  // ordinal allocation and vm to EmitC conversion to prevent constant folding
  // of the tests during the lattter. This means we would need to add a pattern
  // that inserts calls to `iree_vm_ref_retain` for operand/result pairs of the
  // barrier op.
  vm.export @test_ref_eq attributes {emitc.exclude}
  vm.func @test_ref_eq() {
    %ref_1 = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_1_dno = vm.optimization_barrier %ref_1 : !vm.buffer
    %ref_2 = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_2_dno = vm.optimization_barrier %ref_2 : !vm.buffer
    vm.check.eq %ref_1_dno, %ref_2_dno : !vm.buffer
    vm.return
  }

  vm.export @test_ref_ne
  vm.func @test_ref_ne() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    vm.check.ne %ref_a_dno, %ref_b_dno : !vm.buffer
    vm.return
  }

  vm.export @test_ref_nz
  vm.func @test_ref_nz() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.check.nz %ref_dno : !vm.buffer
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Ref lifetime through calls
  // These tests verify refs survive being passed to callees.
  //===--------------------------------------------------------------------===//

  // Pass ref to callee, verify caller's ref is still valid after return.
  vm.export @test_ref_survives_call attributes {emitc.exclude}
  vm.func @test_ref_survives_call() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.check.nz %ref_dno, "ref valid before call" : !vm.buffer
    vm.call @_consume_ref(%ref_dno) : (!vm.buffer) -> ()
    // Ref should still be valid after the call.
    vm.check.nz %ref_dno, "ref valid after call" : !vm.buffer
    vm.return
  }

  vm.func private @_consume_ref(%arg : !vm.buffer)
      attributes {inlining_policy = #util.inline.never} {
    %arg_dno = vm.optimization_barrier %arg : !vm.buffer
    vm.check.nz %arg_dno, "ref valid in callee" : !vm.buffer
    vm.return
  }

  // Pass same ref multiple times to same call.
  vm.export @test_same_ref_multiple_args attributes {emitc.exclude}
  vm.func @test_same_ref_multiple_args() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.call @_consume_two_refs(%ref_dno, %ref_dno) : (!vm.buffer, !vm.buffer) -> ()
    // Ref should still be valid after the call.
    vm.check.nz %ref_dno, "ref valid after call with same ref twice" : !vm.buffer
    vm.return
  }

  vm.func private @_consume_two_refs(%arg0 : !vm.buffer, %arg1 : !vm.buffer)
      attributes {inlining_policy = #util.inline.never} {
    %arg0_dno = vm.optimization_barrier %arg0 : !vm.buffer
    %arg1_dno = vm.optimization_barrier %arg1 : !vm.buffer
    vm.check.nz %arg0_dno, "first arg valid" : !vm.buffer
    vm.check.nz %arg1_dno, "second arg valid" : !vm.buffer
    vm.check.eq %arg0_dno, %arg1_dno, "both args are same ref" : !vm.buffer
    vm.return
  }

  // Return ref from callee, verify it's valid in caller.
  vm.export @test_ref_returned_from_call attributes {emitc.exclude}
  vm.func @test_ref_returned_from_call() {
    %ref = vm.call @_return_ref() : () -> !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.check.nz %ref_dno, "returned ref is valid" : !vm.buffer
    vm.return
  }

  vm.func private @_return_ref() -> !vm.buffer
      attributes {inlining_policy = #util.inline.never} {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    vm.return %ref : !vm.buffer
  }

  // Pass ref, callee returns the same ref.
  vm.export @test_ref_passthrough attributes {emitc.exclude}
  vm.func @test_ref_passthrough() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %returned = vm.call @_passthrough_ref(%ref_dno) : (!vm.buffer) -> !vm.buffer
    %returned_dno = vm.optimization_barrier %returned : !vm.buffer
    vm.check.eq %ref_dno, %returned_dno, "passthrough returns same ref" : !vm.buffer
    vm.return
  }

  vm.func private @_passthrough_ref(%arg : !vm.buffer) -> !vm.buffer
      attributes {inlining_policy = #util.inline.never} {
    vm.return %arg : !vm.buffer
  }

  //===--------------------------------------------------------------------===//
  // Ref lifetime in control flow
  //===--------------------------------------------------------------------===//

  // Ref passed to both branches of cond_br.
  vm.export @test_ref_cond_br_both_paths attributes {emitc.exclude}
  vm.func @test_ref_cond_br_both_paths() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %c1 = vm.const.i32 1
    %c1_dno = vm.optimization_barrier %c1 : i32
    vm.cond_br %c1_dno, ^bb1(%ref_dno : !vm.buffer), ^bb2(%ref_dno : !vm.buffer)
  ^bb1(%arg1 : !vm.buffer):
    vm.check.nz %arg1, "ref valid in bb1" : !vm.buffer
    vm.check.eq %arg1, %ref_dno, "bb1 got same ref" : !vm.buffer
    vm.return
  ^bb2(%arg2 : !vm.buffer):
    vm.check.nz %arg2, "ref valid in bb2" : !vm.buffer
    vm.check.eq %arg2, %ref_dno, "bb2 got same ref" : !vm.buffer
    vm.return
  }

  // Ref passed to one branch, not the other.
  vm.export @test_ref_cond_br_one_path attributes {emitc.exclude}
  vm.func @test_ref_cond_br_one_path() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %c1 = vm.const.i32 1
    %c1_dno = vm.optimization_barrier %c1 : i32
    vm.cond_br %c1_dno, ^bb1(%ref_dno : !vm.buffer), ^bb2
  ^bb1(%arg1 : !vm.buffer):
    vm.check.nz %arg1, "ref valid in bb1" : !vm.buffer
    vm.return
  ^bb2:
    vm.return
  }

  // Ref used in loop body (back-edge liveness).
  vm.export @test_ref_in_loop attributes {emitc.exclude}
  vm.func @test_ref_in_loop() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c3 = vm.const.i32 3
    vm.br ^loop(%c0, %ref_dno : i32, !vm.buffer)
  ^loop(%i : i32, %loop_ref : !vm.buffer):
    // Verify ref is valid on each iteration.
    vm.check.nz %loop_ref, "ref valid in loop" : !vm.buffer
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %c3 : i32
    vm.cond_br %cmp, ^loop(%i_next, %loop_ref : i32, !vm.buffer), ^exit
  ^exit:
    vm.return
  }

  // Multiple different refs through loop.
  vm.export @test_multiple_refs_in_loop attributes {emitc.exclude}
  vm.func @test_multiple_refs_in_loop() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c3 = vm.const.i32 3
    vm.br ^loop(%c0, %ref_a_dno, %ref_b_dno : i32, !vm.buffer, !vm.buffer)
  ^loop(%i : i32, %loop_ref_a : !vm.buffer, %loop_ref_b : !vm.buffer):
    vm.check.nz %loop_ref_a, "ref_a valid in loop" : !vm.buffer
    vm.check.nz %loop_ref_b, "ref_b valid in loop" : !vm.buffer
    vm.check.ne %loop_ref_a, %loop_ref_b, "refs are different" : !vm.buffer
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %c3 : i32
    vm.cond_br %cmp, ^loop(%i_next, %loop_ref_a, %loop_ref_b : i32, !vm.buffer, !vm.buffer), ^exit
  ^exit:
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Ref with globals
  //===--------------------------------------------------------------------===//

  // Store ref to global, load back, verify equality.
  vm.export @test_global_store_load_ref attributes {emitc.exclude}
  vm.func @test_global_store_load_ref() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.global.store.ref %ref_dno, @global_ref : !vm.buffer
    %loaded = vm.global.load.ref @global_ref : !vm.buffer
    %loaded_dno = vm.optimization_barrier %loaded : !vm.buffer
    vm.check.eq %ref_dno, %loaded_dno, "loaded ref equals stored ref" : !vm.buffer
    vm.return
  }

  // Store ref to global, use original ref after.
  vm.export @test_ref_valid_after_global_store attributes {emitc.exclude}
  vm.func @test_ref_valid_after_global_store() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.check.nz %ref_dno, "ref valid before store" : !vm.buffer
    vm.global.store.ref %ref_dno, @global_ref : !vm.buffer
    // Original ref should still be valid after storing to global.
    vm.check.nz %ref_dno, "ref valid after store to global" : !vm.buffer
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Ref with lists
  //===--------------------------------------------------------------------===//

  // Store ref in list, retrieve, verify equality.
  // Uses variant list (!vm.list<?>) since typed lists require type registration.
  vm.export @test_list_set_get_ref attributes {emitc.exclude}
  vm.func @test_list_set_get_ref() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<?>
    vm.list.resize %list, %c1 : (!vm.list<?>, i32)
    vm.list.set.ref %list, %c0, %ref_dno : (!vm.list<?>, i32, !vm.buffer)
    %retrieved = vm.list.get.ref %list, %c0 : (!vm.list<?>, i32) -> !vm.buffer
    %retrieved_dno = vm.optimization_barrier %retrieved : !vm.buffer
    vm.check.eq %ref_dno, %retrieved_dno, "retrieved ref equals set ref" : !vm.buffer
    vm.return
  }

  // Multiple refs in same list.
  vm.export @test_list_multiple_refs attributes {emitc.exclude}
  vm.func @test_list_multiple_refs() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %list = vm.list.alloc %c2 : (i32) -> !vm.list<?>
    vm.list.resize %list, %c2 : (!vm.list<?>, i32)
    vm.list.set.ref %list, %c0, %ref_a_dno : (!vm.list<?>, i32, !vm.buffer)
    vm.list.set.ref %list, %c1, %ref_b_dno : (!vm.list<?>, i32, !vm.buffer)
    %retrieved_a = vm.list.get.ref %list, %c0 : (!vm.list<?>, i32) -> !vm.buffer
    %retrieved_a_dno = vm.optimization_barrier %retrieved_a : !vm.buffer
    %retrieved_b = vm.list.get.ref %list, %c1 : (!vm.list<?>, i32) -> !vm.buffer
    %retrieved_b_dno = vm.optimization_barrier %retrieved_b : !vm.buffer
    vm.check.eq %ref_a_dno, %retrieved_a_dno, "retrieved ref_a equals set ref_a" : !vm.buffer
    vm.check.eq %ref_b_dno, %retrieved_b_dno, "retrieved ref_b equals set ref_b" : !vm.buffer
    vm.check.ne %retrieved_a_dno, %retrieved_b_dno, "refs are different" : !vm.buffer
    vm.return
  }

  // Get ref from list, use multiple times.
  vm.export @test_list_get_ref_multiple_uses attributes {emitc.exclude}
  vm.func @test_list_get_ref_multiple_uses() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<?>
    vm.list.resize %list, %c1 : (!vm.list<?>, i32)
    vm.list.set.ref %list, %c0, %ref_dno : (!vm.list<?>, i32, !vm.buffer)
    %retrieved = vm.list.get.ref %list, %c0 : (!vm.list<?>, i32) -> !vm.buffer
    %retrieved_dno = vm.optimization_barrier %retrieved : !vm.buffer
    // Use retrieved ref multiple times.
    vm.check.nz %retrieved_dno, "retrieved ref valid (use 1)" : !vm.buffer
    vm.check.nz %retrieved_dno, "retrieved ref valid (use 2)" : !vm.buffer
    vm.check.eq %ref_dno, %retrieved_dno, "retrieved ref equals original" : !vm.buffer
    vm.return
  }

  // Ref survives after being stored in list.
  vm.export @test_ref_valid_after_list_set attributes {emitc.exclude}
  vm.func @test_ref_valid_after_list_set() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.check.nz %ref_dno, "ref valid before list set" : !vm.buffer
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<?>
    vm.list.resize %list, %c1 : (!vm.list<?>, i32)
    vm.list.set.ref %list, %c0, %ref_dno : (!vm.list<?>, i32, !vm.buffer)
    // Original ref should still be valid.
    vm.check.nz %ref_dno, "ref valid after list set" : !vm.buffer
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Select operations
  //===--------------------------------------------------------------------===//

  // vm.select.ref with both refs valid.
  vm.export @test_select_ref_true attributes {emitc.exclude}
  vm.func @test_select_ref_true() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %c1 = vm.const.i32 1
    %c1_dno = vm.optimization_barrier %c1 : i32
    %result = vm.select.ref %c1_dno, %ref_a_dno, %ref_b_dno : !vm.buffer
    %result_dno = vm.optimization_barrier %result : !vm.buffer
    vm.check.eq %result_dno, %ref_a_dno, "select true returns first ref" : !vm.buffer
    vm.return
  }

  vm.export @test_select_ref_false attributes {emitc.exclude}
  vm.func @test_select_ref_false() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %c0 = vm.const.i32 0
    %c0_dno = vm.optimization_barrier %c0 : i32
    %result = vm.select.ref %c0_dno, %ref_a_dno, %ref_b_dno : !vm.buffer
    %result_dno = vm.optimization_barrier %result : !vm.buffer
    vm.check.eq %result_dno, %ref_b_dno, "select false returns second ref" : !vm.buffer
    vm.return
  }

  // vm.select.ref with one ref used after select.
  vm.export @test_select_ref_input_survives attributes {emitc.exclude}
  vm.func @test_select_ref_input_survives() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %c1 = vm.const.i32 1
    %c1_dno = vm.optimization_barrier %c1 : i32
    %result = vm.select.ref %c1_dno, %ref_a_dno, %ref_b_dno : !vm.buffer
    %result_dno = vm.optimization_barrier %result : !vm.buffer
    // Both input refs should still be valid after select.
    vm.check.nz %ref_a_dno, "ref_a valid after select" : !vm.buffer
    vm.check.nz %ref_b_dno, "ref_b valid after select" : !vm.buffer
    vm.check.nz %result_dno, "result valid" : !vm.buffer
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Complex multi-use patterns
  //===--------------------------------------------------------------------===//

  // Same ref used in multiple operations sequentially.
  vm.export @test_ref_multiple_sequential_uses attributes {emitc.exclude}
  vm.func @test_ref_multiple_sequential_uses() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    // Use 1: check nz
    vm.check.nz %ref_dno, "use 1" : !vm.buffer
    // Use 2: pass to call
    vm.call @_consume_ref(%ref_dno) : (!vm.buffer) -> ()
    // Use 3: check nz again
    vm.check.nz %ref_dno, "use 3" : !vm.buffer
    // Use 4: store to global
    vm.global.store.ref %ref_dno, @global_ref : !vm.buffer
    // Use 5: final check
    vm.check.nz %ref_dno, "use 5" : !vm.buffer
    vm.return
  }

  // Chain of calls passing same ref.
  vm.export @test_ref_call_chain attributes {emitc.exclude}
  vm.func @test_ref_call_chain() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %result = vm.call @_call_chain_a(%ref_dno) : (!vm.buffer) -> !vm.buffer
    %result_dno = vm.optimization_barrier %result : !vm.buffer
    vm.check.eq %ref_dno, %result_dno, "chain returns same ref" : !vm.buffer
    vm.return
  }

  vm.func private @_call_chain_a(%arg : !vm.buffer) -> !vm.buffer
      attributes {inlining_policy = #util.inline.never} {
    %result = vm.call @_call_chain_b(%arg) : (!vm.buffer) -> !vm.buffer
    vm.return %result : !vm.buffer
  }

  vm.func private @_call_chain_b(%arg : !vm.buffer) -> !vm.buffer
      attributes {inlining_policy = #util.inline.never} {
    vm.return %arg : !vm.buffer
  }

  // Return multiple refs.
  vm.export @test_return_multiple_refs attributes {emitc.exclude}
  vm.func @test_return_multiple_refs() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %results:2 = vm.call @_return_two_refs(%ref_a_dno, %ref_b_dno)
        : (!vm.buffer, !vm.buffer) -> (!vm.buffer, !vm.buffer)
    vm.check.eq %results#0, %ref_a_dno, "first result is ref_a" : !vm.buffer
    vm.check.eq %results#1, %ref_b_dno, "second result is ref_b" : !vm.buffer
    vm.return
  }

  vm.func private @_return_two_refs(%a : !vm.buffer, %b : !vm.buffer)
      -> (!vm.buffer, !vm.buffer)
      attributes {inlining_policy = #util.inline.never} {
    vm.return %a, %b : !vm.buffer, !vm.buffer
  }

  // Return refs in swapped order.
  vm.export @test_return_refs_swapped attributes {emitc.exclude}
  vm.func @test_return_refs_swapped() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %results:2 = vm.call @_return_refs_swapped(%ref_a_dno, %ref_b_dno)
        : (!vm.buffer, !vm.buffer) -> (!vm.buffer, !vm.buffer)
    vm.check.eq %results#0, %ref_b_dno, "first result is ref_b (swapped)" : !vm.buffer
    vm.check.eq %results#1, %ref_a_dno, "second result is ref_a (swapped)" : !vm.buffer
    vm.return
  }

  vm.func private @_return_refs_swapped(%a : !vm.buffer, %b : !vm.buffer)
      -> (!vm.buffer, !vm.buffer)
      attributes {inlining_policy = #util.inline.never} {
    vm.return %b, %a : !vm.buffer, !vm.buffer
  }

  //===--------------------------------------------------------------------===//
  // Discard refs operations
  //===--------------------------------------------------------------------===//

  // Discard zero refs (no-op but should not crash).
  vm.export @test_discard_empty attributes {emitc.exclude}
  vm.func private @test_discard_empty() {
    vm.discard.refs
    vm.return
  }

  // Discard a single ref.
  vm.export @test_discard_single_ref attributes {emitc.exclude}
  vm.func private @test_discard_single_ref() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    vm.check.nz %ref_dno, "ref valid before discard" : !vm.buffer
    vm.discard.refs %ref_dno : !vm.buffer
    // Note: After discard, the ref is released. We shouldn't use it.
    vm.return
  }

  // Discard multiple refs at once.
  vm.export @test_discard_multiple_refs attributes {emitc.exclude}
  vm.func private @test_discard_multiple_refs() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    vm.check.nz %ref_a_dno, "ref_a valid before discard" : !vm.buffer
    vm.check.nz %ref_b_dno, "ref_b valid before discard" : !vm.buffer
    vm.discard.refs %ref_a_dno, %ref_b_dno : !vm.buffer, !vm.buffer
    vm.return
  }

  // Discard in a branch - verify control flow works.
  vm.export @test_discard_in_branch attributes {emitc.exclude}
  vm.func private @test_discard_in_branch() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %c1 = vm.const.i32 1
    %c1_dno = vm.optimization_barrier %c1 : i32
    vm.cond_br %c1_dno, ^bb1, ^bb2
  ^bb1:
    vm.discard.refs %ref_dno : !vm.buffer
    vm.br ^exit
  ^bb2:
    // Don't discard on this path.
    vm.br ^exit
  ^exit:
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Edge case: Nested loops with refs
  //===--------------------------------------------------------------------===//

  // Outer loop carries ref, inner loop doesn't touch it.
  vm.export @test_nested_loop_outer_ref attributes {emitc.exclude}
  vm.func private @test_nested_loop_outer_ref() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    vm.br ^outer(%c0, %ref_dno : i32, !vm.buffer)
  ^outer(%outer_i : i32, %outer_ref : !vm.buffer):
    // Verify ref valid on each outer iteration.
    vm.check.nz %outer_ref, "ref valid in outer loop" : !vm.buffer
    vm.br ^inner(%c0 : i32)
  ^inner(%inner_i : i32):
    // Inner loop doesn't touch ref.
    %inner_next = vm.add.i32 %inner_i, %c1 : i32
    %inner_cmp = vm.cmp.lt.i32.s %inner_next, %c2 : i32
    vm.cond_br %inner_cmp, ^inner(%inner_next : i32), ^outer_check
  ^outer_check:
    %outer_next = vm.add.i32 %outer_i, %c1 : i32
    %outer_cmp = vm.cmp.lt.i32.s %outer_next, %c2 : i32
    vm.cond_br %outer_cmp, ^outer(%outer_next, %outer_ref : i32, !vm.buffer), ^exit
  ^exit:
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Edge case: Ref swap (ping-pong) in loop
  //===--------------------------------------------------------------------===//

  // Loop swaps two refs on each iteration.
  vm.export @test_ping_pong_swap attributes {emitc.exclude}
  vm.func private @test_ping_pong_swap() {
    %ref_a = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_a_dno = vm.optimization_barrier %ref_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @buffer_b : !vm.buffer
    %ref_b_dno = vm.optimization_barrier %ref_b : !vm.buffer
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c3 = vm.const.i32 3
    vm.br ^loop(%c0, %ref_a_dno, %ref_b_dno : i32, !vm.buffer, !vm.buffer)
  ^loop(%i : i32, %x : !vm.buffer, %y : !vm.buffer):
    // Verify both refs valid.
    vm.check.nz %x, "x valid in loop" : !vm.buffer
    vm.check.nz %y, "y valid in loop" : !vm.buffer
    vm.check.ne %x, %y, "x and y are different" : !vm.buffer
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %c3 : i32
    // SWAP: x->y_arg, y->x_arg
    vm.cond_br %cmp, ^loop(%i_next, %y, %x : i32, !vm.buffer, !vm.buffer), ^exit
  ^exit:
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Edge case: Diamond with asymmetric use
  //===--------------------------------------------------------------------===//

  // One path uses the ref, other doesn't. Both should work.
  vm.export @test_diamond_asymmetric_use attributes {emitc.exclude}
  vm.func private @test_diamond_asymmetric_use() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %c1 = vm.const.i32 1
    %c1_dno = vm.optimization_barrier %c1 : i32
    vm.cond_br %c1_dno, ^use_path(%ref_dno : !vm.buffer), ^nouse_path(%ref_dno : !vm.buffer)
  ^use_path(%r1 : !vm.buffer):
    vm.check.nz %r1, "ref valid in use_path" : !vm.buffer
    vm.br ^merge
  ^nouse_path(%r2 : !vm.buffer):
    // Don't use r2 - just forward.
    vm.br ^merge
  ^merge:
    vm.return
  }

  // Same test but with condition false - takes nouse_path.
  vm.export @test_diamond_asymmetric_nouse attributes {emitc.exclude}
  vm.func private @test_diamond_asymmetric_nouse() {
    %ref = vm.const.ref.rodata @buffer_a : !vm.buffer
    %ref_dno = vm.optimization_barrier %ref : !vm.buffer
    %c0 = vm.const.i32 0
    %c0_dno = vm.optimization_barrier %c0 : i32
    vm.cond_br %c0_dno, ^use_path(%ref_dno : !vm.buffer), ^nouse_path(%ref_dno : !vm.buffer)
  ^use_path(%r1 : !vm.buffer):
    vm.check.nz %r1, "ref valid in use_path" : !vm.buffer
    vm.br ^merge
  ^nouse_path(%r2 : !vm.buffer):
    // Don't use r2 - just forward. Should still be released properly.
    vm.br ^merge
  ^merge:
    vm.return
  }

}
