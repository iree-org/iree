// RUN: iree-compile %s \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic \
// RUN:   --iree-hal-indirect-command-buffers=false \
// RUN:   -o %t.vmfb
// RUN: iree-run-module \
// RUN:   --device=local-sync \
// RUN:   --module=%t.vmfb \
// RUN:   --function=main \
// RUN:   --input=7xi32=0 \
// RUN:   --input=7xi32=0 \
// RUN:   --input=7xi32=0 \
// RUN:   --input=i32=0 \
// RUN:   --input=i32=0 \
// RUN:   --input=i1=00 \
// RUN:   --input=i32=2 \
// RUN:   --expected_output=7xi32=1,1,1,1,1,1,1 \
// RUN:   --expected_output=7xi32=0,0,0,0,0,0,0 \
// RUN:   --expected_output=7xi32=1,1,1,1,1,1,1 \
// RUN:   --expected_output=i32=0 \
// RUN:   --expected_output=i32=0 \
// RUN:   --expected_output=i1=00
// RUN: iree-run-module \
// RUN:   --device=local-sync \
// RUN:   --module=%t.vmfb \
// RUN:   --function=main \
// RUN:   --input=7xi32=0 \
// RUN:   --input=7xi32=0 \
// RUN:   --input=7xi32=0 \
// RUN:   --input=i32=0 \
// RUN:   --input=i32=0 \
// RUN:   --input=i1=01 \
// RUN:   --input=i32=2 \
// RUN:   --expected_output=7xi32=1,1,1,1,1,1,1 \
// RUN:   --expected_output=7xi32=0,0,0,0,0,0,0 \
// RUN:   --expected_output=7xi32=1,1,1,1,1,1,1 \
// RUN:   --expected_output=i32=0 \
// RUN:   --expected_output=i32=0 \
// RUN:   --expected_output=i1=01

// Regression test for https://github.com/iree-org/iree/issues/24753.
// With indirect command buffers disabled, both selector paths must execute
// successfully and preserve every returned value.
module {
  func.func public @main(
      %arg0: tensor<7xi32>, %arg1: tensor<7xi32>, %arg2: tensor<7xi32>,
      %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i1>,
      %arg6: tensor<i32>)
      -> (tensor<7xi32>, tensor<7xi32>, tensor<7xi32>, tensor<i32>,
          tensor<i32>, tensor<i1>) {
    %false = stablehlo.constant dense<false> : tensor<i1>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %is_zero = stablehlo.compare EQ, %arg6, %c0, SIGNED
        : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %case_index = stablehlo.convert %is_zero
        : (tensor<i1>) -> tensor<i32>
    %0:4 = "stablehlo.case"(%case_index) ({
      %is_one = stablehlo.compare EQ, %arg6, %c1, SIGNED
          : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %nested_case_index = stablehlo.convert %is_one
          : (tensor<i1>) -> tensor<i32>
      %1:4 = "stablehlo.case"(%nested_case_index) ({
        %splat1_a = stablehlo.broadcast_in_dim %c1, dims = []
            : (tensor<i32>) -> tensor<7xi32>
        %result0 = stablehlo.add %arg0, %splat1_a : tensor<7xi32>
        %splat1_b = stablehlo.broadcast_in_dim %c1, dims = []
            : (tensor<i32>) -> tensor<7xi32>
        %result2 = stablehlo.add %arg2, %splat1_b : tensor<7xi32>
        %not_arg5 = stablehlo.not %arg5 : tensor<i1>
        %inner_case_index = stablehlo.convert %not_arg5
            : (tensor<i1>) -> tensor<i32>
        %result3 = "stablehlo.case"(%inner_case_index) ({
          %splat0 = stablehlo.broadcast_in_dim %c0, dims = []
              : (tensor<i32>) -> tensor<7xi32>
          %positive = stablehlo.compare GT, %result0, %splat0, SIGNED
              : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi1>
          %any_positive = stablehlo.reduce(%positive init: %false)
              applies stablehlo.or across dimensions = [0]
              : (tensor<7xi1>, tensor<i1>) -> tensor<i1>
          %sum_case_index = stablehlo.convert %any_positive
              : (tensor<i1>) -> tensor<i32>
          %selected = "stablehlo.case"(%sum_case_index) ({
            stablehlo.return %arg3 : tensor<i32>
          }, {
            %sum = stablehlo.reduce(%arg1 init: %c0)
                applies stablehlo.add across dimensions = [0]
                : (tensor<7xi32>, tensor<i32>) -> tensor<i32>
            stablehlo.return %sum : tensor<i32>
          }) : (tensor<i32>) -> tensor<i32>
          stablehlo.return %selected : tensor<i32>
        }, {
          stablehlo.return %arg3 : tensor<i32>
        }) : (tensor<i32>) -> tensor<i32>
        stablehlo.return %result0, %result2, %result3, %c0
            : tensor<7xi32>, tensor<7xi32>, tensor<i32>, tensor<i32>
      }, {
        stablehlo.return %arg0, %arg2, %arg3, %arg4
            : tensor<7xi32>, tensor<7xi32>, tensor<i32>, tensor<i32>
      }) : (tensor<i32>) ->
          (tensor<7xi32>, tensor<7xi32>, tensor<i32>, tensor<i32>)
      stablehlo.return %1#0, %1#1, %1#2, %1#3
          : tensor<7xi32>, tensor<7xi32>, tensor<i32>, tensor<i32>
    }, {
      %not_arg5 = stablehlo.not %arg5 : tensor<i1>
      %inner_case_index = stablehlo.convert %not_arg5
          : (tensor<i1>) -> tensor<i32>
      %result3 = "stablehlo.case"(%inner_case_index) ({
        %splat0 = stablehlo.broadcast_in_dim %c0, dims = []
            : (tensor<i32>) -> tensor<7xi32>
        %positive = stablehlo.compare GT, %arg0, %splat0, SIGNED
            : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi1>
        %any_positive = stablehlo.reduce(%positive init: %false)
            applies stablehlo.or across dimensions = [0]
            : (tensor<7xi1>, tensor<i1>) -> tensor<i1>
        %sum_case_index = stablehlo.convert %any_positive
            : (tensor<i1>) -> tensor<i32>
        %selected = "stablehlo.case"(%sum_case_index) ({
          stablehlo.return %arg3 : tensor<i32>
        }, {
          %sum = stablehlo.reduce(%arg1 init: %c0)
              applies stablehlo.add across dimensions = [0]
              : (tensor<7xi32>, tensor<i32>) -> tensor<i32>
          stablehlo.return %sum : tensor<i32>
        }) : (tensor<i32>) -> tensor<i32>
        stablehlo.return %selected : tensor<i32>
      }, {
        stablehlo.return %arg3 : tensor<i32>
      }) : (tensor<i32>) -> tensor<i32>
      stablehlo.return %arg0, %arg2, %result3, %arg4
          : tensor<7xi32>, tensor<7xi32>, tensor<i32>, tensor<i32>
    }) : (tensor<i32>) ->
        (tensor<7xi32>, tensor<7xi32>, tensor<i32>, tensor<i32>)
    return %0#0, %arg1, %0#1, %0#2, %0#3, %arg5
        : tensor<7xi32>, tensor<7xi32>, tensor<7xi32>, tensor<i32>,
          tensor<i32>, tensor<i1>
  }
}
