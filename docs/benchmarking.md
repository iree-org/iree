# Benchmarking

IREE uses benchmarks to inspect performance at varying levels of granularity.
Benchmarking is implemented using the
[Google Benchmark library](https://github.com/google/benchmark) and tracing with
C++ bindings from the
[Google Web Tracing Framework](https://github.com/google/tracing-framework).

## Module Benchmarks

`iree-benchmark-module` is a program accepting (almost) the same inputs as
`iree-run-module` that will benchmark the invocation of a single entry function.
It measures timing for the whole process of invoking a function through the VM,
including allocating and freeing output buffers. This is a high-level benchmark
of an entire invocation flow. It provides a big picture view, but depends on
many different variables, like an integration test. For finer-grained
measurements more akin to unit tests, see [Microbenchmarks](#microbenchmarks)
and [Tracing](#tracing).

To use `iree-benchmark-module` generate an IREE module for the target backend:

```shell
$ bazel run //iree/tools:iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=interpreter-bytecode \
  $PWD/iree/tools/test/simple.mlir \
  -o /tmp/module.fb
```

and then benchmark an exported function in that module:

```shell
$ bazel run //iree/tools:iree-benchmark-module -- \
  --input_file=/tmp/module.fb \
  --driver=interpreter \
  --entry_function=abs \
  --inputs="i32=-2"
```

You'll see output like

```shell
Run on (12 X 4500 MHz CPU s)
CPU Caches:
  L1 Data 32K (x6)
  L1 Instruction 32K (x6)
  L2 Unified 1024K (x6)
  L3 Unified 8448K (x1)
Load Average: 2.21, 1.93, 3.34
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may
 be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_RunModule/process_time/real_time     218193 ns       231884 ns         3356
```

Notice that there are a few warnings in there (you may not see all of these).
The benchmark library helpfully warns about some common issues that will affect
benchmark timing. When trying to obtain real benchmark numbers, you should
generally build an optimized build (`-c opt` in Bazel) and disable CPU scaling.

Another thing to consider is that depending on where you are running the
benchmark you might want to avoid additional programs running at the same time.
Bazel itself runs a server even when it's not being actively invoked that can be
quite a memory hog, so we'll instead invoke the binary directly. First make sure
that you've built an optimized binary.

```shell
$ bazel build -c opt //iree/tools:iree-benchmark-module
```

Disable CPU scaling. On Linux, benchmark provides some
[instructions](https://github.com/google/benchmark#disabling-cpu-frequency-scaling):

Use your favorite process manager (e.g. [htop](https://hisham.hm/htop/) or
[pkill](https://en.wikipedia.org/wiki/Pkill) on Linux) to kill heavy-weight
programs such as Chrome and Bazel.

```shell
$ sudo cpupower frequency-set --governor performance
```

TODO(scotttodd): Windows instructions

Now we'll actually invoke the binary:

```shell
$ ./bazel-bin/iree/tools/iree-benchmark-module \
  --input_file=/tmp/module.fb \
  --driver=interpreter \
  --entry_function=abs \
  --inputs="i32=-2"
```

```shell
Run on (12 X 4500 MHz CPU s)
CPU Caches:
  L1 Data 32K (x6)
  L1 Instruction 32K (x6)
  L2 Unified 1024K (x6)
  L3 Unified 8448K (x1)
Load Average: 1.49, 3.42, 3.49
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_RunModule/process_time/real_time      11416 ns        14202 ns        61654
```

Remember to restore cpu scaling when you're done:

```shell
$ sudo cpupower frequency-set --governor powersave
```

## Microbenchmarks

We also benchmark the performance of individual parts (more of these coming
soon) of the IREE system in isolation. These measurements provide more targeted
metrics to direct development work.

### Bytecode Module Benchmarks

TODO(benvanik): Talk about VM Benchmarks

## Tracing

IREE is instrumented with the C++ bindings from the
[Google Web Tracing Framework](https://github.com/google/tracing-framework).

TODO(benvanik): Talk about WTF
