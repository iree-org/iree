func.func @double_index(%arg0 : index) -> index {
  %0 = arith.addi %arg0, %arg0 : index
  return %0 : index
}

func.func @index_times_four(%arg0 : index) -> index {
  %0 = arith.addi %arg0, %arg0 : index
  %1 = arith.addi %0, %0 : index
  return %1 : index
}
