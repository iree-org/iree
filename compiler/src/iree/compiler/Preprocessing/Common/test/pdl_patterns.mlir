pdl.pattern @arith_optimization : benefit(1) {

  %f32_type = pdl.type : f32
  %zero = pdl.operand
  %one = pdl.operand
  %two = pdl.operand
  %three = pdl.operand

  %a_op = pdl.operation "arith.divf" (%zero, %one : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
  %a = pdl.result 0 of %a_op

  %b_op = pdl.operation "arith.divf" (%two, %three : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
  %b = pdl.result 0 of %b_op

  %mul = pdl.operation "arith.mulf" (%a, %b : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)

  pdl.rewrite %mul {
    %a_new_op = pdl.operation "arith.mulf" (%zero, %two : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
    %a_new = pdl.result 0 of %a_new_op
    %b_new_op = pdl.operation "arith.mulf" (%one, %three : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
    %b_new = pdl.result 0 of %b_new_op
    %div = pdl.operation "arith.divf" (%a_new, %b_new : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
    pdl.replace %mul with %div
  }
}
