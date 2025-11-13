This directory contains examples to get developers started
with using IREE's HalModuleDebugSink feature with callbacks
to bisect layers of multi-dispatch model, to locate numerical
divergences. It contains 2 examples:

## Simple callback example: `sync_callback_log.py`

A simple multi-dispatch example showing how to log statistics of
all dispatch inputs and outputs. When a model is compiled with
`--iree-flow-trace-dispatch-tensors`, IREE inserts trace points
before and after each dispatch function.

When iree-run-module executes a model with trace points, the
tensor values of dispatch inputs and outputs are printed to standard
output.

For models with small tensors, this is useful for debugging
numerical issues: By comparing tensor values before and after a
numerical regression, one can often find the dispatch that
introduced the numerical error. However, for large models, it is
infeasible to print all tensor values to the terminal. Even saving
tensor values to disk may be impractical due to size constraints.

In such cases, it is often sufficient to compare simple summary
statistics, such as the mean. To enable this level of flexibility,
one can use HalModuleDebugSink with arbitrary callback functions.
This example presents a simple callback that logs mean values.

## Two model callback example: `sync_callback_log_async.py`

This example extends Example 1. It shows how to run two versions of
a model in parallel. The versions differ only in one dispatch, and
the goal is automatically identify the 'bad' dispatch.

In practice, the two models are usually compilation artifacts of
the same source model, built before and after a numerical
regression. This technique is especially useful when numerical
divergence appears early in execution.
