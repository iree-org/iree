// RUN: iree-opt --split-input-file --verify-diagnostics --iree-vm-conversion %s

// -----
// expected-error @+1 {{conversion to vm.module failed}}
module @negative_half_add {
  module {
    func.func @run(%x: f16, %y: f16) -> f16 {
      // expected-error @+1 {{failed to legalize operation 'arith.addf'}}
      %result = arith.addf %x, %y : f16
      return %result : f16
    }
  }
}

// -----
// expected-error @+1 {{conversion to vm.module failed}}
module @negative_bfloat_negate {
  module {
    func.func @run(%x: bf16, %y: bf16) -> bf16 {
      // expected-error @+1 {{failed to legalize operation 'arith.negf'}}
      %result = arith.negf %x : bf16
      return %result : bf16
    }
  }
}

// -----
// expected-error @+1 {{conversion to vm.module failed}}
module @negative_half_sqrt {
  module {
    func.func @run(%x: f16, %y: f16) -> f16 {
      // expected-error @+1 {{failed to legalize operation 'math.sqrt'}}
      %result = math.sqrt %x : f16
      return %result : f16
    }
  }
}

// -----
// expected-error @+1 {{conversion to vm.module failed}}
module @negative_bfloat_pow {
  module {
    func.func @run(%x: bf16, %y: bf16) -> bf16 {
      // expected-error @+1 {{failed to legalize operation 'math.powf'}}
      %result = math.powf %x, %y : bf16
      return %result : bf16
    }
  }
}

// -----
// expected-error @+1 {{conversion to vm.module failed}}
module @negative_half_fma {
  module {
    func.func @run(%x: f16, %y: f16) -> f16 {
      // expected-error @+1 {{failed to legalize operation 'math.fma'}}
      %result = math.fma %x, %y, %x : f16
      return %result : f16
    }
  }
}
