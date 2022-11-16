// Make sure that *not all* vm arithmetic ops are (unconditionally) speculatable.

// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(vm.module(loop-invariant-code-motion))" %s | \
// RUN:   FileCheck %s

// CHECK-LABEL: @no_speculate_integer
vm.module @no_speculate_integer {
  // CHECK-LABEL: vm.func @add_i32
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.add.i32
  vm.func @add_i32(%arg0: i32, %arg1: i32,
                   %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.add.i32 %arg0, %arg1 : i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @mul_i32
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.mul.i32
  vm.func @mul_i32(%arg0: i32, %arg1: i32,
                   %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.mul.i32 %arg0, %arg1 : i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @div_ui32
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.div.i32.u
  vm.func @div_ui32(%arg0: i32, %arg1: i32,
                    %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.div.i32.u %arg0, %arg1 : i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @shl_i32
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.shl.i32
  vm.func @shl_i32(%arg0: i32, %arg1: i32,
                   %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.shl.i32 %arg0, %arg1 : i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @fma_i64
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      vm.fma.i64
  vm.func @fma_i64(%arg0: i64, %arg1: i64, %arg2: i64,
                   %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.fma.i64 %arg0, %arg1, %arg2 : i64
    }
    vm.return
  }
}

// -----

// CHECK-LABEL: @speculate_integer
vm.module @speculate_integer {
  // CHECK-LABEL: vm.func @const_i32
  // CHECK-NEXT:    vm.const.i32 0
  vm.func @const_i32(%lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.const.i32 0
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @not_i32
  // CHECK-NEXT:    vm.not.i32
  vm.func @not_i32(%arg0: i32, %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.not.i32 %arg0 : i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @and_i32
  // CHECK-NEXT:    vm.and.i32
  vm.func @and_i32(%arg0: i32, %arg1: i32,
                   %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.and.i32 %arg0, %arg1 : i32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @switch_i64
  // CHECK-NEXT:    vm.switch.i64
  vm.func @switch_i64(%arg0: i32, %arg1: i64, %arg2: i64, %arg3: i64,
                      %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.switch.i64 %arg0[%arg1, %arg2] else %arg3 : i64
    }
    vm.return
  }
} // end module

// -----

// CHECK-LABEL: @speculate_float
vm.module @speculate_float {
  // CHECK-LABEL: vm.func @add_f64
  // CHECK-NEXT:    vm.add.f64
  vm.func @add_f64(%arg0: f64, %arg1: f64,
                   %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.add.f64 %arg0, %arg1 : f64
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @mul_f64
  // CHECK-NEXT:    vm.mul.f64
  vm.func @mul_f64(%arg0: f64, %arg1: f64,
                   %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.mul.f64 %arg0, %arg1 : f64
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @neg_f64
  // CHECK-NEXT:    vm.neg.f64
  vm.func @neg_f64(%arg0: f64, %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.neg.f64 %arg0 : f64
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @atan_f32
  // CHECK-NEXT:    vm.atan.f32
  vm.func @atan_f32(%arg0: f32, %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.atan.f32 %arg0 : f32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @switch_f64
  // CHECK-NEXT:    vm.switch.f64
  vm.func @switch_f64(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: f64,
                      %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.switch.f64 %arg0[%arg1, %arg2] else %arg3 : f64
    }
    vm.return
  }
}

// -----

// CHECK-LABEL: @speculate_casts
vm.module @speculate_casts {
  // CHECK-LABEL: vm.func @cast_ui32_f32
  // CHECK-NEXT:    vm.cast.ui32.f32
  vm.func @cast_ui32_f32(%arg0: i32,
                         %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.cast.ui32.f32 %arg0 : i32 -> f32
    }
    vm.return
  }

  // CHECK-LABEL: vm.func @bitcast_f32_i32
  // CHECK-NEXT:    vm.bitcast.f32.i32
  vm.func @bitcast_f32_i32(%arg0: f32, %arg1: i32,
                           %lb: index, %ub: index, %step: index) -> () {
    scf.for %i = %lb to %ub step %step {
      %val = vm.bitcast.f32.i32 %arg0 : f32 -> i32
    }
    vm.return
  }
}
