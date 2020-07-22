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

To use `iree-benchmark-module`, generate an IREE module for the target backend:

```shell
$ bazel run //iree/tools:iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vmla \
  $PWD/iree/tools/test/simple.mlir \
  -o /tmp/module.fb
```

and then benchmark an exported function in that module:

```shell
$ bazel run //iree/tools:iree-benchmark-module -- \
  --input_file=/tmp/module.fb \
  --driver=vmla \
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
BM_RunModule/process_time/real_time       0.22 ms         0.23 ms         3356
```

Notice that there are a few warnings in there (you may not see all of these).
The benchmark library helpfully warns about some common issues that will affect
benchmark timing. When trying to obtain real benchmark numbers, you should
generally build an optimized build (`-c opt` in Bazel) and
[disable CPU scaling](#cpu-configuration).

```shell
$ bazel build -c opt //iree/tools:iree-benchmark-module
```

Another thing to consider is that depending on where you are running the
benchmark you might want to avoid additional programs running at the same time.
Bazel itself runs a server even when it's not being actively invoked that can be
quite a memory hog, so we'll instead invoke the binary directly. Use your
favorite process manager (e.g. [htop](https://hisham.hm/htop/) or
[pkill](https://en.wikipedia.org/wiki/Pkill) on Linux) to kill heavy-weight
programs such as Chrome and Bazel.

Now we'll actually invoke the binary:

```shell
$ ./bazel-bin/iree/tools/iree-benchmark-module \
  --input_file=/tmp/module.fb \
  --driver=vmla \
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
BM_RunModule/process_time/real_time      0.011 ms        0.014 ms        61654
```

Remember to [restore CPU scaling](#cpu-configuration) when you're done.

## Microbenchmarks

We also benchmark the performance of individual parts (more of these coming
soon) of the IREE system in isolation. These measurements provide more targeted
metrics to direct development work.

### Bytecode Module Benchmarks

TODO(benvanik): Talk about VM Benchmarks

### Building Tracy

To use tracing in IREE, you need to build IREE with following requirements:

*   Turn `IREE_ENABLE_RUNTIME_TRACING` on.
*   Add `-DNDEBUG` to `IREE_DEFAULT_COPTS`.
*   Use Release/RelWithDebInfo build.

For example:

```shell
export IREE_DEFAULT_COPTS='-DNDEBUG'
cmake -B build/ \
      -DIREE_ENABLE_RUNTIME_TRACING=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

The above compiles IREE with Tracy APIs so that IREE will stream profiling data
back to Tracy when running. To be able to collect and analyze these data, you
can either use GUI or CLI tools. Tracy profiler is the GUI tool. You can find
the
[Tracy manual](https://github.com/wolfpld/tracy/releases/download/v0.6.3/tracy.pdf)
for more details on Tracy itself.

To build the profiler on Linux, you will need to install some external
libraries. Some Linux distributions will require you to add a `lib` prefix and a
`-dev`, or `-devel` postfix to library names. For example, you might see the
error:

```
Package glfw3 was not found in the pkg-config search path.

```

and then you could try to install `libglfw3-dev`.

Instructions to build Tracy profiler:

```shell
cd third_party/tracy/profiler/build/unix
make release
```

### Using Tracy

Launch the profiler UI, and click connect. Then the server will wait for the
connection. Now you can launch the IREE binary you want to trace, it should
connect automatically and stream data. For example:

Prepare the module to profile:

```shell
build/iree/tools/iree-benchmark-module \
  --input_file=/tmp/module.fb \
  --driver=vmla \
  --entry_function=abs \
  --inputs="i32=-2"
```

Run the module:

```shell
build/iree/tools/iree-run-module \
  --input_file=/tmp/module.fb \
  --driver=vmla \
  --entry_function=abs \
  --inputs="i32=-2"
```

## CPU Configuration

When benchmarking, it's important to consider the configuration of your CPUs.
Most notably, CPU scaling can give variable results, so you'll usually want to
disable it. This can get pretty complex, but the most basic thing to do is to
run all CPUs at maximum frequency.

### Linux

Google benchmark provides some
[instructions](https://github.com/google/benchmark#disabling-cpu-frequency-scaling):

Turn off CPU scaling before benchmarking:

```shell
$ sudo cpupower frequency-set --governor performance
```

Restore CPU scaling after benchmarking:

```shell
$ sudo cpupower frequency-set --governor powersave
```

### Android

Android doesn't give us quite as nice tooling, but the principle is basically
the same. You will likely need to be root (use `su` or `adb root`). The commands
will depend on your exact phone and number of cores. First play around and make
sure you understand what everything means.

Some useful commands:

```shell
$ cat /proc/cpuinfo
$ cat /sys/devices/system/cpu/possible
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
$ cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
$ cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq
$ cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
$ cat /sys/devices/system/cpu/cpu0/cpufreq/affected_cpus
$ cat /sys/devices/system/cpu/cpu0/online
```

One common case is if you want to set the quota governor of 8 CPUs for
performance. Make sure to check their current settings first so you can put them
back when you're done.

```shell
$ for i in `seq 0 7`; do cat "/sys/devices/system/cpu/cpu${i?}/cpufreq/scaling_governor"; done
```

```shell
$ for i in `seq 0 7`; do echo performance > "/sys/devices/system/cpu/cpu${i?}/cpufreq/scaling_governor"; done
```

and then double check that all CPUs are now at their maximum frequency

```shell
$ for i in `seq 0 7`; do paste "/sys/devices/system/cpu/cpu${i?}/cpufreq/cpuinfo_cur_freq" "/sys/devices/system/cpu/cpu${i?}/cpufreq/cpuinfo_max_freq"; done
```

TODO(scotttodd): Windows instructions
