TensorList is a module in IREE, which means that it hooks into IREE's conversion
flow. There are two main conversions that happen during IREE compilation, which
have corresponding directories in `iree/compiler/Dialect/HAL/Conversion/`,
namely FlowToHAL and HALToVM.

For our purposes here, FlowToHAL does the following type conversions:

-   converts from `tensor<....>` to `!hal.buffer_view`

-   converts from the frontend representation of TensorList (e.g.
    `!tf_tensorlist.list`) to `!tensorlist.list` (this dialect).

For our purposes here, HALToVM maps ops to VM-level calls into a custom module.

This dialect hooks into those conversion processes. Unfortunately, the FlowToHAL
step requires taking a dependency on tf_tensorlist which depends on TensorFlow.
We don't want to take a dependency on TensorFlow here, so we put the FlowToHAL
step in the `tf_tensorlist` dialect.
