# Using Address/Memory/Thread sanitizers

## Enabling the sanitizers

In the CMake build system of IREE, at least on Linux and Android, enabling these sanitizers is a simple matter of passing one of these options to the initial CMake command:

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

These symbolizers will be most helpful on builds with debug info, so consider using

```
-DCMAKE_BUILD_TYPE=RelWithDebInfo
```

instead of just `Release`. It's also fine to use sanitizers on `Debug` builds, of course --- if the issue that you're tracking down reproduces at all in a debug build! Sanitizers are often used to track down subtle issues that may only manifest themselves in certain build configurations.

## No ThreadSanitizer on Android

ThreadSanitizer is currently considered unimplemented on Android, see [this](https://github.com/android/ndk/issues/1171) NDK feature request.

It was apparently / sort-of working in NDK r19c so if you need it badly, consider trying going back to it. It doesn't build at all on NDK r21d.

## Symbolizing Android stacks

On desktop platforms, sanitizer builds normally output a conveniently symbolized stack right away.

Not so on Android, due to [this](https://github.com/android/ndk/issues/753) Android NDK issue. Your Android sanitizer builds will only output unsymbolized (i.e. cryptic!) stacks, like this:
```
    #0 0x7ad2d85b4c  (/system/lib64/libclang_rt.asan-aarch64-android.so+0x9fb4c)
    #1 0x653f64ec10  (/data/local/tmp/iree-benchmark-module+0x178c10)
    #2 0x653f64d658  (/data/local/tmp/iree-benchmark-module+0x177658)
```

Copy this raw output from the sanitizer and feed it into the `stdin` of the `scripts/android_symbolize.sh` script, with the `ANDROID_NDK` environment variable pointing to the NDK root directory, like this:

```shell
ANDROID_NDK=~/android-ndk-r21d ./scripts/android_symbolize.sh < /tmp/asan.txt
```

Where `/tmp/asan.txt` is where you've pasted the raw sanitizer report.

You need to have set the `ANDROID_NDK` environment variable to point to the NDK root directory.

**Tip:** this script will happily just echo any line that isn't a stack frame. That means you can feed it the whole `ASan` report at once, and it will output a symbolized version of it. DO NOT run it on a single stack at a time! That is unlike the symbolizer tool that's being added in NDK r22, and one of the reasons why we prefer to keep our own script. For more details see [this comment](https://github.com/android/ndk/issues/753#issuecomment-719719789)
