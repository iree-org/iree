// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=64},cse)' %s | FileCheck %s --check-prefixes=CHECK,INDEX64
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=32},cse)' %s | FileCheck %s --check-prefixes=CHECK,INDEX32

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

// -----

// Narrow operands must be normalized independently of their producers. Equality
// and unsigned ordering use zero extension; signed ordering uses sign extension.
// CHECK-LABEL: vm.module public @cmp_i8 {
module @cmp_i8 {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i32, %[[RHS:[^:]+]]: i32)
  func.func @fn(%lhs: i8, %rhs: i8) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
    // CHECK: %[[LHS_U:.+]] = vm.ext.i8.i32.u %[[LHS]] : i32 -> i32
    // CHECK: %[[RHS_U:.+]] = vm.ext.i8.i32.u %[[RHS]] : i32 -> i32
    // CHECK: %[[EQ:.+]] = vm.cmp.eq.i32 %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[NE:.+]] = vm.cmp.ne.i32 %[[LHS_U]], %[[RHS_U]] : i32
    %eq = arith.cmpi eq, %lhs, %rhs : i8
    %ne = arith.cmpi ne, %lhs, %rhs : i8
    // CHECK: %[[LHS_S:.+]] = vm.ext.i8.i32.s %[[LHS]] : i32 -> i32
    // CHECK: %[[RHS_S:.+]] = vm.ext.i8.i32.s %[[RHS]] : i32 -> i32
    // CHECK: %[[SLT:.+]] = vm.cmp.lt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    // CHECK: %[[SLE:.+]] = vm.cmp.lte.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    // CHECK: %[[SGT:.+]] = vm.cmp.gt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    // CHECK: %[[SGE:.+]] = vm.cmp.gte.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    %slt = arith.cmpi slt, %lhs, %rhs : i8
    %sle = arith.cmpi sle, %lhs, %rhs : i8
    %sgt = arith.cmpi sgt, %lhs, %rhs : i8
    %sge = arith.cmpi sge, %lhs, %rhs : i8
    // CHECK: %[[ULT:.+]] = vm.cmp.lt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[ULE:.+]] = vm.cmp.lte.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[UGT:.+]] = vm.cmp.gt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[UGE:.+]] = vm.cmp.gte.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    %ult = arith.cmpi ult, %lhs, %rhs : i8
    %ule = arith.cmpi ule, %lhs, %rhs : i8
    %ugt = arith.cmpi ugt, %lhs, %rhs : i8
    %uge = arith.cmpi uge, %lhs, %rhs : i8
    // CHECK: vm.return %[[EQ]], %[[NE]], %[[SLT]], %[[SLE]], %[[SGT]], %[[SGE]], %[[ULT]], %[[ULE]], %[[UGT]], %[[UGE]] : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
    return %eq, %ne, %slt, %sle, %sgt, %sge, %ult, %ule, %ugt, %uge : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  }
}

// -----

// CHECK-LABEL: vm.module public @cmp_i16 {
module @cmp_i16 {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i32, %[[RHS:[^:]+]]: i32)
  func.func @fn(%lhs: i16, %rhs: i16) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
    // CHECK: %[[LHS_U:.+]] = vm.ext.i16.i32.u %[[LHS]] : i32 -> i32
    // CHECK: %[[RHS_U:.+]] = vm.ext.i16.i32.u %[[RHS]] : i32 -> i32
    // CHECK: %[[EQ:.+]] = vm.cmp.eq.i32 %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[NE:.+]] = vm.cmp.ne.i32 %[[LHS_U]], %[[RHS_U]] : i32
    %eq = arith.cmpi eq, %lhs, %rhs : i16
    %ne = arith.cmpi ne, %lhs, %rhs : i16
    // CHECK: %[[LHS_S:.+]] = vm.ext.i16.i32.s %[[LHS]] : i32 -> i32
    // CHECK: %[[RHS_S:.+]] = vm.ext.i16.i32.s %[[RHS]] : i32 -> i32
    // CHECK: %[[SLT:.+]] = vm.cmp.lt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    // CHECK: %[[SLE:.+]] = vm.cmp.lte.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    // CHECK: %[[SGT:.+]] = vm.cmp.gt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    // CHECK: %[[SGE:.+]] = vm.cmp.gte.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    %slt = arith.cmpi slt, %lhs, %rhs : i16
    %sle = arith.cmpi sle, %lhs, %rhs : i16
    %sgt = arith.cmpi sgt, %lhs, %rhs : i16
    %sge = arith.cmpi sge, %lhs, %rhs : i16
    // CHECK: %[[ULT:.+]] = vm.cmp.lt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[ULE:.+]] = vm.cmp.lte.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[UGT:.+]] = vm.cmp.gt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[UGE:.+]] = vm.cmp.gte.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    %ult = arith.cmpi ult, %lhs, %rhs : i16
    %ule = arith.cmpi ule, %lhs, %rhs : i16
    %ugt = arith.cmpi ugt, %lhs, %rhs : i16
    %uge = arith.cmpi uge, %lhs, %rhs : i16
    // CHECK: vm.return %[[EQ]], %[[NE]], %[[SLT]], %[[SLE]], %[[SGT]], %[[SGE]], %[[ULT]], %[[ULE]], %[[UGT]], %[[UGE]] : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
    return %eq, %ne, %slt, %sle, %sgt, %sge, %ult, %ule, %ugt, %uge : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  }
}

// -----

// Signed i1 true is -1. Shifts also discard high bits from narrow arithmetic.
// CHECK-LABEL: vm.module public @cmp_i1 {
module @cmp_i1 {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i32, %[[RHS:[^:]+]]: i32)
  func.func @fn(%lhs: i1, %rhs: i1) -> (i1, i1, i1) {
    // CHECK: %[[SHIFT:.+]] = vm.const.i32 31
    // CHECK: %[[LHS_SHIFTED:.+]] = vm.shl.i32 %[[LHS]], %[[SHIFT]] : i32
    // CHECK: %[[LHS_S:.+]] = vm.shr.i32.s %[[LHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_SHIFTED:.+]] = vm.shl.i32 %[[RHS]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_S:.+]] = vm.shr.i32.s %[[RHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[SLT:.+]] = vm.cmp.lt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    %slt = arith.cmpi slt, %lhs, %rhs : i1
    // CHECK: %[[MASK:.+]] = vm.const.i32 1
    // CHECK: %[[LHS_U:.+]] = vm.and.i32 %[[LHS]], %[[MASK]] : i32
    // CHECK: %[[RHS_U:.+]] = vm.and.i32 %[[RHS]], %[[MASK]] : i32
    // CHECK: %[[ULT:.+]] = vm.cmp.lt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    // CHECK: %[[EQ:.+]] = vm.cmp.eq.i32 %[[LHS_U]], %[[RHS_U]] : i32
    %ult = arith.cmpi ult, %lhs, %rhs : i1
    %eq = arith.cmpi eq, %lhs, %rhs : i1
    // CHECK: vm.return %[[SLT]], %[[ULT]], %[[EQ]] : i32, i32, i32
    return %slt, %ult, %eq : i1, i1, i1
  }
}

// -----

// CHECK-LABEL: vm.module public @cmp_i2 {
module @cmp_i2 {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i32, %[[RHS:[^:]+]]: i32)
  func.func @fn(%lhs: i2, %rhs: i2) -> (i1, i1) {
    // CHECK: %[[SHIFT:.+]] = vm.const.i32 30
    // CHECK: %[[LHS_SHIFTED:.+]] = vm.shl.i32 %[[LHS]], %[[SHIFT]] : i32
    // CHECK: %[[LHS_S:.+]] = vm.shr.i32.s %[[LHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_SHIFTED:.+]] = vm.shl.i32 %[[RHS]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_S:.+]] = vm.shr.i32.s %[[RHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[SLT:.+]] = vm.cmp.lt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    %slt = arith.cmpi slt, %lhs, %rhs : i2
    // CHECK: %[[MASK:.+]] = vm.const.i32 3
    // CHECK: %[[LHS_U:.+]] = vm.and.i32 %[[LHS]], %[[MASK]] : i32
    // CHECK: %[[RHS_U:.+]] = vm.and.i32 %[[RHS]], %[[MASK]] : i32
    // CHECK: %[[ULT:.+]] = vm.cmp.lt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    %ult = arith.cmpi ult, %lhs, %rhs : i2
    // CHECK: vm.return %[[SLT]], %[[ULT]] : i32, i32
    return %slt, %ult : i1, i1
  }
}

// -----

// CHECK-LABEL: vm.module public @cmp_i4 {
module @cmp_i4 {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i32, %[[RHS:[^:]+]]: i32)
  func.func @fn(%lhs: i4, %rhs: i4) -> (i1, i1) {
    // CHECK: %[[SHIFT:.+]] = vm.const.i32 28
    // CHECK: %[[LHS_SHIFTED:.+]] = vm.shl.i32 %[[LHS]], %[[SHIFT]] : i32
    // CHECK: %[[LHS_S:.+]] = vm.shr.i32.s %[[LHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_SHIFTED:.+]] = vm.shl.i32 %[[RHS]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_S:.+]] = vm.shr.i32.s %[[RHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[SLT:.+]] = vm.cmp.lt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    %slt = arith.cmpi slt, %lhs, %rhs : i4
    // CHECK: %[[MASK:.+]] = vm.const.i32 15
    // CHECK: %[[LHS_U:.+]] = vm.and.i32 %[[LHS]], %[[MASK]] : i32
    // CHECK: %[[RHS_U:.+]] = vm.and.i32 %[[RHS]], %[[MASK]] : i32
    // CHECK: %[[ULT:.+]] = vm.cmp.lt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    %ult = arith.cmpi ult, %lhs, %rhs : i4
    // CHECK: vm.return %[[SLT]], %[[ULT]] : i32, i32
    return %slt, %ult : i1, i1
  }
}

// -----

// The type converter also accepts non-power-of-two widths.
// CHECK-LABEL: vm.module public @cmp_i7 {
module @cmp_i7 {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i32, %[[RHS:[^:]+]]: i32)
  func.func @fn(%lhs: i7, %rhs: i7) -> (i1, i1) {
    // CHECK: %[[SHIFT:.+]] = vm.const.i32 25
    // CHECK: %[[LHS_SHIFTED:.+]] = vm.shl.i32 %[[LHS]], %[[SHIFT]] : i32
    // CHECK: %[[LHS_S:.+]] = vm.shr.i32.s %[[LHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_SHIFTED:.+]] = vm.shl.i32 %[[RHS]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_S:.+]] = vm.shr.i32.s %[[RHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[SLT:.+]] = vm.cmp.lt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    %slt = arith.cmpi slt, %lhs, %rhs : i7
    // CHECK: %[[MASK:.+]] = vm.const.i32 127
    // CHECK: %[[LHS_U:.+]] = vm.and.i32 %[[LHS]], %[[MASK]] : i32
    // CHECK: %[[RHS_U:.+]] = vm.and.i32 %[[RHS]], %[[MASK]] : i32
    // CHECK: %[[ULT:.+]] = vm.cmp.lt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    %ult = arith.cmpi ult, %lhs, %rhs : i7
    // CHECK: vm.return %[[SLT]], %[[ULT]] : i32, i32
    return %slt, %ult : i1, i1
  }
}

// -----

// CHECK-LABEL: vm.module public @cmp_i31 {
module @cmp_i31 {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i32, %[[RHS:[^:]+]]: i32)
  func.func @fn(%lhs: i31, %rhs: i31) -> (i1, i1) {
    // CHECK: %[[SHIFT:.+]] = vm.const.i32 1
    // CHECK: %[[LHS_SHIFTED:.+]] = vm.shl.i32 %[[LHS]], %[[SHIFT]] : i32
    // CHECK: %[[LHS_S:.+]] = vm.shr.i32.s %[[LHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_SHIFTED:.+]] = vm.shl.i32 %[[RHS]], %[[SHIFT]] : i32
    // CHECK: %[[RHS_S:.+]] = vm.shr.i32.s %[[RHS_SHIFTED]], %[[SHIFT]] : i32
    // CHECK: %[[SLT:.+]] = vm.cmp.lt.i32.s %[[LHS_S]], %[[RHS_S]] : i32
    %slt = arith.cmpi slt, %lhs, %rhs : i31
    // CHECK: %[[MASK:.+]] = vm.const.i32 2147483647
    // CHECK: %[[LHS_U:.+]] = vm.and.i32 %[[LHS]], %[[MASK]] : i32
    // CHECK: %[[RHS_U:.+]] = vm.and.i32 %[[RHS]], %[[MASK]] : i32
    // CHECK: %[[ULT:.+]] = vm.cmp.lt.i32.u %[[LHS_U]], %[[RHS_U]] : i32
    %ult = arith.cmpi ult, %lhs, %rhs : i31
    // CHECK: vm.return %[[SLT]], %[[ULT]] : i32, i32
    return %slt, %ult : i1, i1
  }
}

// -----

// index is not a narrow IntegerType, even with a 32-bit VM index representation.
// CHECK-LABEL: vm.module public @cmp_index {
module @cmp_index {
  // CHECK: vm.func private @fn(%[[LHS:[^:]+]]: i{{32|64}}, %[[RHS:[^:]+]]: i{{32|64}})
  func.func @fn(%lhs: index, %rhs: index) -> i1 {
    // INDEX32-NEXT: %[[RESULT:.+]] = vm.cmp.lt.i32.s %[[LHS]], %[[RHS]] : i32
    // INDEX64-NEXT: %[[RESULT:.+]] = vm.cmp.lt.i64.s %[[LHS]], %[[RHS]] : i64
    // CHECK-NEXT: vm.return %[[RESULT]] : i32
    %result = arith.cmpi slt, %lhs, %rhs : index
    return %result : i1
  }
}
