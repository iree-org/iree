---
layout: default
permalink: developing-iree/sanitizers
title: "Using Address/Memory/Thread Sanitizers"
parent: Developing IREE
---

# Using Address/Memory/Thread Sanitizers
{: .no_toc }

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

## Enabling the sanitizers

In the CMake build system of IREE, at least on Linux and Android, enabling these
sanitizers is a simple matter of passing one of these options to the initial
CMake command:

```
-DIREE_ENABLE_ASAN=ON
```

or

```
-DIREE_ENABLE_MSAN=ON
```

or

```
-DIREE_ENABLE_TSAN=ON
```

These sanitizers will be most helpful on builds with debug info, so consider
using

```
-DCMAKE_BUILD_TYPE=RelWithDebInfo
```

instead of just `Release`. It's also fine to use sanitizers on `Debug` builds,
of course --- if the issue that you're tracking down reproduces at all in a
debug build! Sanitizers are often used to track down subtle issues that may only
manifest themselves in certain build configurations.

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
`scripts/android_symbolize.sh` script, with the `ANDROID_NDK` environment
variable pointing to the NDK root directory, like this:

```shell
ANDROID_NDK=~/android-ndk-r21d ./scripts/android_symbolize.sh < /tmp/asan.txt
```

Where `/tmp/asan.txt` is where you've pasted the raw sanitizer report.

**Tip:** this script will happily just echo any line that isn't a stack frame.
That means you can feed it the whole `ASan` report at once, and it will output a
symbolized version of it. DO NOT run it on a single stack at a time! That is
unlike the symbolizer tool that's being added in NDK r22, and one of the reasons
why we prefer to keep our own script. For more details see [this
comment](https://github.com/android/ndk/issues/753#issuecomment-719719789)