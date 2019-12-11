// RUN: iree-run-mlir %s --target_backends=interpreter-bytecode --input_values="i32=42" --output_types="i,i,i,i,i,i,i,i,i,i" | IreeFileCheck %s

// CHECK-LABEL: EXEC @cmpi
func @cmpi(%42 : i32) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) { // need at least one arg to avoid constant folding
  %cm1 = constant -1 : i32
  %eq = cmpi "eq", %42, %cm1 : i32
  %ne = cmpi "ne", %42, %cm1 : i32
  %slt = cmpi "slt", %42, %cm1 : i32
  %sle = cmpi "sle", %42, %cm1 : i32
  %sgt = cmpi "sgt", %42, %cm1 : i32
  %sge = cmpi "sge", %42, %cm1 : i32
  %ult = cmpi "ult", %42, %cm1 : i32
  %ule = cmpi "ule", %42, %cm1 : i32
  %ugt = cmpi "ugt", %42, %cm1 : i32
  %uge = cmpi "uge", %42, %cm1 : i32
  return %eq, %ne, %slt, %sle, %sgt, %sge, %ult, %ule, %ugt, %uge : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}
// CHECK: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=0
