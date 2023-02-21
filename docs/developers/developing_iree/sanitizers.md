# Using Address/Memory/Thread Sanitizers

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
ASan   | Out-of-bounds accesses,<br>Use-after-free,<br>Use-after-return,<br>Memory leaks (*), ... | Crashes,<br>non-deterministic results,<br>memory leaks (*) | 2x | 3x | Yes
MSan   | Uninitialized memory reads | Non-deterministic results | 3x | ? | Yes
TSan   | Data races | Many bugs in multi-thread code | 5x-15x | 5x-10x | [No](https://github.com/android/ndk/issues/1171)

Notes:
* (*) See [this
  documentation](https://clang.llvm.org/docs/AddressSanitizer.html#memory-leak-detection)
  on leak detection. It is only enabled by default on some platforms.

## Support status and how to enable each sanitizer

### ASan (AddressSanitizer)

Enabling ASan in the IREE build is a simple matter of setting the
`IREE_ENABLE_ASAN` CMake option:

```
cmake -DIREE_ENABLE_ASAN=ON ...
```

### TSan (ThreadSanitizer)

To enable TSan, at the moment, the following 3 CMake options must be set:

```
cmake \
  -DIREE_ENABLE_TSAN=ON \
  -DIREE_BYTECODE_MODULE_ENABLE_TSAN=ON \
  -DIREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON \
  -DIREE_BUILD_SAMPLES=OFF \
  ...
```

In practice, `IREE_ENABLE_TSAN` alone would be enough for many targets, but not
all. The problem is that a IREE runtime built with `IREE_ENABLE_TSAN` cannot
load a IREE compiled LLVM/CPU module unless the following flags were passed to
the IREE compiler: `--iree-llvm-sanitize=thread` and
`--iree-llvm-link-embedded=false`.

The CMake options `IREE_BYTECODE_MODULE_ENABLE_TSAN` and
`IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER` ensure that the above flags are
passed to the IREE compiler when building modules used in tests, benchmarks,
etc. (anything that internally uses the CMake `iree_bytecode_module` macro).

The CMake option `IREE_BUILD_SAMPLES=OFF` is needed because samples [currently
assume](https://github.com/openxla/iree/pull/8893) that the embedded linker is
used, so they are incompatible with
`IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON`.

At the moment, CMake logic heavy-handedly enforces that whenever
`IREE_ENABLE_TSAN` is set, these other two CMake variables are also set.
That ensures that all tests succeed: no test is expected to fail with TSan.

If you know what you're doing (i.e. if you are not building targets that
internally involve a LLVM/CPU `iree_bytecode_module`), feel free to locally comment out
the CMake error and only set `IREE_ENABLE_TSAN`. Also see a
[past attempt]((https://github.com/openxla/iree/pull/8966) to relax that CMake
validation.

### MSan (MemorySanitizer)

In theory that should be a simple matter of

```
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

```
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
`build_tools/scripts/android_symbolize.sh` script, with the `ANDROID_NDK` environment
variable pointing to the NDK root directory, like this:

```shell
ANDROID_NDK=~/android-ndk-r21d ./build_tools/scripts/android_symbolize.sh < /tmp/asan.txt
```

Where `/tmp/asan.txt` is where you've pasted the raw sanitizer report.

**Tip:** this script will happily just echo any line that isn't a stack frame.
That means you can feed it the whole `ASan` report at once, and it will output a
symbolized version of it. DO NOT run it on a single stack at a time! That is
unlike the symbolizer tool that's being added in NDK r22, and one of the reasons
why we prefer to keep our own script. For more details see [this
comment](https://github.com/android/ndk/issues/753#issuecomment-719719789)
