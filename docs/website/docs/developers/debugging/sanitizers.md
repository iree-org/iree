---
icon: material/broom
---

# Sanitizers (ASan/MSan/TSan)

[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html),
[MemorySanitizer](https://clang.llvm.org/docs/MemorySanitizer.html) and
[ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) are tools
provided by `clang` to detect certain classes of errors in C/C++ programs. They
consist of compiler instrumentation (so your program's executable code is
modified) and runtime libraries (so e.g. the `malloc` function may get
replaced).

They are abbreviated as "ASan", "MSan" and "TSan" respectively.

They all incur large overhead, so only enable them while debugging.

Tool   | Detects | Helps debug what? | Slowdown | Memory overhead | Android support
------ | ------- | ----------------- | -------- | --------------- | ---------------
ASan   | Out-of-bounds accesses, use-after-free, use-after-return, memory leaks | Crashes, non-deterministic results, memory leaks | 2x | 3x | Yes
MSan   | Uninitialized memory reads | Non-deterministic results | 3x | ? | Yes
TSan   | Data races | Many bugs in multi-thread code | 5x-15x | 5x-10x | [No](https://github.com/android/ndk/issues/1171)

!!! note

    See
    [this documentation](https://clang.llvm.org/docs/AddressSanitizer.html#memory-leak-detection)
    on leak detection. It is only enabled by default on some platforms.

## Support status and how to enable each sanitizer

### ASan (AddressSanitizer)

To enable ASan:

```shell
cmake -DIREE_ENABLE_ASAN=ON ...
```

Several `_asan` tests like
`iree/tests/e2e/stablehlo_ops/check_llvm-cpu_local-task_asan_abs.mlir` are
also defined when using this configuration. These tests include AddressSanitizer
in compiled CPU code as well by using these `iree-compile` flags:

```shell
--iree-llvmcpu-link-embedded=false
--iree-llvmcpu-sanitize=address
```

#### Linking to the dynamic ASan runtime

You may want to use ASan when using the python bindings.
One way to achieve this is to build Python (or whatever executable that is
going to use IREE as a shared library) with Asan.
Another option is to link to the ASan runtime dynamically instead of
linking it statically into an executable.

Using clang-12 (other versions should also work) as a example, configure IREE
with something like:

```shell
cmake \
  -DIREE_ENABLE_ASAN=ON \
  -DCMAKE_EXE_LINKER_FLAGS=-shared-libasan \
  -DCMAKE_SHARED_LINKER_FLAGS=-shared-libasan \
  -DCMAKE_C_COMPILER=clang-12 \
  -DCMAKE_CXX_COMPILER=clang++-12 \
  ...
```

Then when running things the ASan runtime will have to be preloaded.

```shell
LD_PRELOAD=/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so \
ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-12/bin/llvm-symbolizer \
  python ...
```

On Ubuntu the corresponding ASan runtime is provided by a package like
`libclang-common-12-dev` depending on your Clang version.
E.g.

```shell
sudo apt install libclang-common-12-dev llvm-12 clang-12
```

Note that during building would also need to preload the ASan runtime, since
the build executes its own binaries that are linked against the runtime.

```shell
LD_PRELOAD=/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so \
ASAN_OPTIONS=detect_leaks=0 \
ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-12/bin/llvm-symbolizer \
  cmake --build ...
```

!!! tip

    If you want to run the IREE CUDA runtime driver it is likely you would
    need.

    ```shell
    ASAN_OPTIONS="protect_shadow_gap=0"
    ```

    Like this

    ```shell
    LD_PRELOAD=/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so \
    ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-12/bin/llvm-symbolizer \
    ASAN_OPTIONS="protect_shadow_gap=0" \
      python ...
    ```

### TSan (ThreadSanitizer)

To enable TSan:

```shell
cmake -DIREE_ENABLE_TSAN=ON ...
```

Several `_tsan` tests like
`iree/tests/e2e/stablehlo_ops/check_llvm-cpu_local-task_tsan_abs.mlir` are
also defined when using this configuration. These tests include ThreadSanitizer
in compiled CPU code as well by using these `iree-compile` flags:

```shell
--iree-llvmcpu-link-embedded=false
--iree-llvmcpu-sanitize=address
```

Note that a IREE runtime built with TSan cannot load a IREE compiled LLVM/CPU
module unless those flags are used, so other tests are excluded using the
`notsan` label.

### MSan (MemorySanitizer)

In theory that should be a simple matter of

```shell
-DIREE_ENABLE_MSAN=ON
```

However, that requires making and using a custom
build of libc++ with MSan as explained in
[this documentation](https://github.com/google/sanitizers/wiki/MemorySanitizerLibcxxHowTo).

As of April 2022, all of IREE's tests succeeded with MSan on Linux/x86-64,
provided that the `vulkan` driver was disabled (due to lack of MSan
instrumentation in the NVIDIA Vulkan driver).

### UBSan (UndefinedBehaviorSanitizer)

Enabling UBSan in the IREE build is a simple matter of setting the
`IREE_ENABLE_UBSAN` CMake option:

```shell
cmake -DIREE_ENABLE_UBSAN=ON ...
```

Note that both ASan and UBSan can be enabled in the same build.

## Symbolizing the reports

### Desktop platforms

On desktop platforms, getting nicely symbolized reports is covered in [this
documentation](https://clang.llvm.org/docs/AddressSanitizer.html#symbolizing-the-reports).
The gist of it is make sure that `llvm-symbolizer` is in your `PATH`, or make
the `ASAN_SYMBOLIZER_PATH` environment variable point to it.

### Android

On Android it's more complicated due to
[this](https://github.com/android/ndk/issues/753) Android NDK issue.
Fortunately, we have a script to perform the symbolization. Copy the raw output
from the sanitizer and feed it into the `stdin` of the
`build_tools/scripts/android_symbolize.sh` script, with the `ANDROID_NDK`
environment variable pointing to the NDK root directory, like this:

```shell
ANDROID_NDK=~/android-ndk-r21d ./build_tools/scripts/android_symbolize.sh < /tmp/asan.txt
```

Where `/tmp/asan.txt` is where you've pasted the raw sanitizer report.

!!! tip

    This script will happily just echo any line that isn't a stack frame.
    That means you can feed it the whole `ASan` report at once, and it will
    output a symbolized version of it. DO NOT run it on a single stack at a
    time! That is unlike the symbolizer tool that's being added in NDK r22, and
    one of the reasons why we prefer to keep our own script. For more details
    see
    [this comment](https://github.com/android/ndk/issues/753#issuecomment-719719789).
