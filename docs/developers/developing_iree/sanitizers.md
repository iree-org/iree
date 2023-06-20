# Using Sanitizers (ASAN, TSAN, etc)

[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) and
[ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html), abbreviated as "ASAN" and "TSAN" respectively, are LLVM instrumentation features helping to detect certain classes of bugs. They are available both via Clang (to build C/C++ code, such as IREE's binaries) and via `iree-compile` (to build IREE bytecode modules). This page discusses both Clang and `iree-compile` usage, and the interactions between the two, since there are consistency requirements between how IREE bytecode modules are instrumented and how the IREE runtime, which loads them, is instrumented.

## Cheat sheet

Sanitizer to use  | ASAN    | TSAN
----------------- | ------------------ | -----------------
Compile my own sanitizer-instrumented bytecode module | `iree-compile --iree-llvmcpu-sanitize=address --iree-llvmcpu-link-embedded=false` . No need for ASAN instrumentation in `iree-compile` itself. | `iree-compile --iree-llvmcpu-sanitize=thread --iree-llvmcpu-link-embedded=false` . No need for TSAN instrumentation in `iree-compile` itself.
Load my own sanitizer-instrumented bytecode module | You can load the module in any IREE runtime, whether or not it's ASAN-instrumented. Enabling ASAN instrumentation in the IREE runtime may yield additional insights but is not necessary. | The IREE runtime must be built with TSAN. Rebuild it with `cmake . -DIREE_ENABLE_TSAN=ON`. Then the resulting IREE runtime tools, e.g. `iree-run-module`, are able to load TSAN-instrumented modules. No need to rebuild the compiler.
Instrument IREE's own binaries (runtime and compiler) | `cmake . -DIREE_ENABLE_ASAN=ON` enables ASAN in all of IREE's C/C++ code (runtime and compiler). | `cmake . -DIREE_ENABLE_TSAN=ON` enables TSAN in all of IREE's C/C++ code (runtime and compiler).
Enable instrumentation in IREE's own tests building bytecode modules. | `cmake . -DIREE_ENABLE_ASAN=ON -DIREE_BYTECODE_MODULE_ENABLE_ASAN=ON -DIREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON` | `cmake . -DIREE_ENABLE_TSAN=ON -DIREE_BYTECODE_MODULE_ENABLE_TSAN=ON -DIREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON`

## Explanation &mdash; The 3 different things that "enabling sanitizers" can mean in IREE

### 1. Telling `iree-compile` to generate sanitizer instrumentation in bytecode modules.

That is achieved by the `--iree-llvmcpu-sanitize={address,thread}` flag, which works similarly to Clang's `-fsanitize=` flag. For example, `iree-compile --iree-llvmcpu-sanitize=address` will generate a bytecode module with ASAN instrumentation.

When passing a `--iree-llvmcpu-sanitize=` flag, one must also pass `--iree-llvmcpu-link-embedded=false`. Sanitizers do not work with the default embedded linker. This flag causes the bytecode module to be linked using the system linker instead.

### 2. Building IREE's own C/C++ code with sanitizer instrumentation.

That is done by the CMake options `IREE_ENABLE_{ASAN,TSAN}`.

This controls all of IREE's C/C++ targets: the runtime, the compiler, the other tools.

TSAN-instrumented bytecode modules can only be loaded by a TSAN-instrumented IREE runtime. By contrast, ASAN-instrumented bytecode modules can be loaded by any IREE runtime.

### 3. Building IREE's test bytecode modules with sanitizer instrumentation.

Many of IREE's tests involve building a bytecode module with `iree-compile`. These a built by the `iree-test-deps` target. To get instrumentation into that module code, the `iree-compile` command used to build test modules must itself pass `--iree-llvmcpu-sanitize={address,thread}`.

That is enabled by the CMake options `IREE_BYTECODE_MODULE_ENABLE_{ASAN,TSAN}`.

As noted above, when `--iree-llvmcpu-sanitize=` is passed, `--iree-llvmcpu-link-embedded=false` must also be passed. Just like `IREE_BYTECODE_MODULE_ENABLE_{ASAN,TSAN}` enables `--iree-llvmcpu-sanitize=` in test modules, `IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER` enables `--iree-llvmcpu-link-embedded=false` in test modules. So, always set `IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER` when setting `IREE_BYTECODE_MODULE_ENABLE_{ASAN,TSAN}`.


## Other sanitizers

Besides the main sanitizers ASAN and TSAN, there is some stub of support for a few additional sanitizers.

### MSAN (MemorySanitizer)

[MSAN](https://clang.llvm.org/docs/MemorySanitizer.html) helps detect use of uninitialized memory.

In theory that should be a simple matter of

```
-DIREE_ENABLE_MSAN=ON
```

However, that requires making and using a custom
build of libc++ with MSAN as explained in
[this documentation](https://github.com/google/sanitizers/wiki/MemorySanitizerLibcxxHowTo).

As of April 2022, all of IREE's tests succeeded with MSAN on Linux/x86-64,
provided that the `vulkan` driver was disabled (due to lack of MSAN
instrumentation in the NVIDIA Vulkan driver).

### UBSan (UndefinedBehaviorSanitizer)

Enabling UBSan in the IREE build is a simple matter of setting the
`IREE_ENABLE_UBSAN` CMake option:

```
cmake -DIREE_ENABLE_UBSAN=ON ...
```

Note that both ASAN and UBSan can be enabled in the same build.

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
`build_tools/scripts/android_symbolize.sh` script, with the `ANDROID_NDK` environment
variable pointing to the NDK root directory, like this:

```shell
ANDROID_NDK=~/android-ndk-r21d ./build_tools/scripts/android_symbolize.sh < /tmp/aSAN.txt
```

Where `/tmp/aSAN.txt` is where you've pasted the raw sanitizer report.

**Tip:** this script will happily just echo any line that isn't a stack frame.
That means you can feed it the whole `ASAN` report at once, and it will output a
symbolized version of it. DO NOT run it on a single stack at a time! That is
unlike the symbolizer tool that's being added in NDK r22, and one of the reasons
why we prefer to keep our own script. For more details see [this
comment](https://github.com/android/ndk/issues/753#issuecomment-719719789)
