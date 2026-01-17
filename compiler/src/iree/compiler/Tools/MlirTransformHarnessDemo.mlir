// Test input for MlirTransformHarnessDemo
func.func @demo_function(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  %1 = arith.muli %0, %arg0 : i32
  return %1 : i32
}
