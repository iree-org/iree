// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-util-verify-structured-control-flow))" --split-input-file --verify-diagnostics %s

// Tests that external functions are ignored.

util.func private @external_func()

// -----

// Tests that a valid structured control flow passes verification.

util.func public @valid_scf_if(%cond: i1) -> i32 {
  %result = scf.if %cond -> i32 {
    %c1 = arith.constant 1 : i32
    scf.yield %c1 : i32
  } else {
    %c2 = arith.constant 2 : i32
    scf.yield %c2 : i32
  }
  util.return %result : i32
}

// -----

// Tests that a valid scf.index_switch passes verification.

util.func public @valid_scf_index_switch(%idx: index) -> i32 {
  %result = scf.index_switch %idx -> i32
  case 0 {
    %c10 = arith.constant 10 : i32
    scf.yield %c10 : i32
  }
  case 1 {
    %c20 = arith.constant 20 : i32
    scf.yield %c20 : i32
  }
  default {
    %c30 = arith.constant 30 : i32
    scf.yield %c30 : i32
  }
  util.return %result : i32
}

// -----

// Tests that a valid single-block scf.execute_region passes verification.

util.func public @valid_scf_execute_region() {
  scf.execute_region {
    %c_true = arith.constant true
    %cond = util.optimization_barrier %c_true : i1
    scf.if %cond {
      %c1 = arith.constant 1 : i32
      util.optimization_barrier %c1 : i32
    } else {
      %c2 = arith.constant 2 : i32
      util.optimization_barrier %c2 : i32
    }
    scf.yield
  }
  util.return
}

// -----

// Tests that nested scf.execute_regions pass verification.

util.func public @nested_scf_execute_region(%cond: i1) {
  scf.if %cond {
    scf.execute_region {
      scf.execute_region {
        %c42 = arith.constant 42 : i32
        util.optimization_barrier %c42 : i32
        scf.yield
      }
      scf.yield
    }
  }
  util.return
}

// -----

// Tests that branch operations in functions trigger an error.
// The presence of cf.br creates a multi-block function with CFG.

util.func public @invalid_branch_in_function() {
  // expected-error @+1 {{unexpected branch operation in function after structured control flow conversion}}
  cf.br ^bb1
^bb1:
  util.return
}

// -----

// Tests that cf.cond_br operations inside scf.execute_regions trigger an error.

util.func public @invalid_cond_br_in_execute_region() {
  scf.execute_region {
    %c_true = arith.constant true
    %cond = util.optimization_barrier %c_true : i1
    // expected-error @+1 {{unexpected branch operation in function after structured control flow conversion}}
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %c1 = arith.constant 1 : i32
    util.optimization_barrier %c1 : i32
    cf.br ^bb3
  ^bb2:
    %c2 = arith.constant 2 : i32
    util.optimization_barrier %c2 : i32
    cf.br ^bb3
  ^bb3:
    scf.yield
  }
  util.return
}
