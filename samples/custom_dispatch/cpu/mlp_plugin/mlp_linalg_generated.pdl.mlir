module {
  pdl.pattern @mlp : benefit(1) {
    %0 = operand loc(#loc2)
    %1 = types loc(#loc3)
    %2 = operation "func.call"(%0 : !pdl.value)  -> (%1 : !pdl.range<type>) loc(#loc3)
    rewrite %2 {
      %3 = attribute = true loc(#loc5)
      %4 = operation "arith.constant"  {"value" = %3} loc(#loc6)
      replace %2 with %4 loc(#loc4)
    } loc(#loc4)
  } loc(#loc1)
} loc(#loc)
#loc = loc("/home/ianwood/iree/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdll":1:1)
#loc1 = loc("/home/ianwood/iree/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdll":5:1)
#loc2 = loc("/home/ianwood/iree/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdll":6:29)
#loc3 = loc("/home/ianwood/iree/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdll":6:14)
#loc4 = loc("/home/ianwood/iree/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdll":7:3)
#loc5 = loc("/home/ianwood/iree/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdll":7:49)
#loc6 = loc("/home/ianwood/iree/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdll":7:21)
