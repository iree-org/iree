// RUN: iree-run-mlir %s --target_backends=interpreter-bytecode --input_values="f32=42.0" --output_types="i,i,i,i,i,i" | FileCheck %s

// CHECK-LABEL: EXEC @cmpf
func @cmpf(%42 : f32) -> (i1, i1, i1, i1, i1, i1) { // need at least one arg to avoid constant folding
  %cm1 = constant -1.0 : f32
  %oeq = cmpf "oeq", %42, %cm1 : f32
  %une = cmpf "une", %42, %cm1 : f32
  %olt = cmpf "olt", %42, %cm1 : f32
  %ole = cmpf "ole", %42, %cm1 : f32
  %ogt = cmpf "ogt", %42, %cm1 : f32
  %oge = cmpf "oge", %42, %cm1 : f32
  return %oeq, %une, %olt, %ole, %ogt, %oge : i1, i1, i1, i1, i1, i1
}
// CHECK: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
