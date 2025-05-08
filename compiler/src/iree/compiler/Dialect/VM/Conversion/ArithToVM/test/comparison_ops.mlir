// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=64},cse)' %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_cmp_eq_i32
module @t001_cmp_eq_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.eq.i32 %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi eq, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t002_cmp_ne_i32
module @t002_cmp_ne_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.ne.i32 %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ne, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t003_cmp_slt_i32
module @t003_cmp_slt_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lt.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi slt, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t004_cmp_sle_i32
module @t004_cmp_sle_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lte.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi sle, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t005_cmp_sgt_i32
module @t005_cmp_sgt_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gt.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi sgt, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t006_cmp_sge_i32
module @t006_cmp_sge_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gte.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi sge, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t007_cmp_ult_i32
module @t007_cmp_ult_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lt.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ult, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t008_cmp_ule_i32
module @t008_cmp_ule_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lte.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ule, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t009_cmp_ugt_i32
module @t009_cmp_ugt_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gt.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ugt, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t010_cmp_uge_i32
module @t010_cmp_uge_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gte.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi uge, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t011_cmp_uge_i64
module @t011_cmp_uge_i64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i64, %arg1 : i64) -> (i1) {
    // CHECK: vm.cmp.gte.i64.u %[[ARG0]], %[[ARG1]] : i64
    %1 = arith.cmpi uge, %arg0, %arg1 : i64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t012_cmp_false_f64
module @t012_cmp_false_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.const.i32.zero
    %1 = arith.cmpf false, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t013_cmp_oeq_f64
module @t013_cmp_oeq_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.eq.f64.o %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf oeq, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t014_cmp_ogt_f64
module @t014_cmp_ogt_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.gt.f64.o %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf ogt, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t015_cmp_oge_f64
module @t015_cmp_oge_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.gte.f64.o %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf oge, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t016_cmp_olt_f64
module @t016_cmp_olt_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.lt.f64.o %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf olt, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t017_cmp_ole_f64
module @t017_cmp_ole_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.lte.f64.o %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf ole, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t018_cmp_one_f64
module @t018_cmp_one_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.ne.f64.o %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf one, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t019_cmp_ord_f64
module @t019_cmp_ord_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK-DAG: %[[ONE:.*]] = vm.const.i32 1
    // CHECK-DAG: %[[ISNAN0:.*]] = vm.cmp.nan.f64 %[[ARG0]]
    // CHECK-DAG: %[[ISNAN1:.*]] = vm.cmp.nan.f64 %[[ARG1]]
    // CHECK-DAG: %[[AND:.*]] = vm.and.i32 %[[ISNAN0]], %[[ISNAN1]]
    // CHECK: %[[XOR:.*]] = vm.xor.i32 %[[ONE]], %[[AND]]
    // CHECK: return %[[XOR]]
    %1 = arith.cmpf ord, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t020_cmp_ueq_f64
module @t020_cmp_ueq_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.eq.f64.u %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf ueq, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t021_cmp_ugt_f64
module @t021_cmp_ugt_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.gt.f64.u %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf ugt, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t022_cmp_uge_f64
module @t022_cmp_uge_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.gte.f64.u %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf uge, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t023_cmp_ult_f64
module @t023_cmp_ult_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.lt.f64.u %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf ult, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t024_cmp_ule_f64
module @t024_cmp_ule_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.lte.f64.u %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf ule, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t025_cmp_une_f64
module @t025_cmp_une_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.cmp.ne.f64.u %[[ARG0]], %[[ARG1]] : f64
    %1 = arith.cmpf une, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t026_cmp_uno_f64
module @t026_cmp_uno_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK-DAG: %[[ISNAN0:.*]] = vm.cmp.nan.f64 %[[ARG0]]
    // CHECK-DAG: %[[ISNAN1:.*]] = vm.cmp.nan.f64 %[[ARG1]]
    // CHECK: %[[OR:.*]] = vm.or.i32 %[[ISNAN0]], %[[ISNAN1]]
    // CHECK: return %[[OR]]
    %1 = arith.cmpf uno, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}


// -----
// CHECK-LABEL: @t027_cmp_true_f64
module @t027_cmp_true_f64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: f64, %arg1 : f64) -> (i1) {
    // CHECK: vm.const.i32 1
    %1 = arith.cmpf true, %arg0, %arg1 : f64
    return %1 : i1
  }
}

}
