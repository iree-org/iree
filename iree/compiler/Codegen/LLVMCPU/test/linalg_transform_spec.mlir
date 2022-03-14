// RUN: iree-opt %s 

pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  tile %0 {sizes = [2]}
}
