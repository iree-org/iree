---
hide:
  - tags
tags:
  - Android
icon: simple/android
---

# Android LLDB debugging

This doc shows how to use [LLDB](https://lldb.llvm.org/index.html) to debug
native binaries on Android. For a more complete explanation, see the
[official LLDB documentation on remote debugging](https://lldb.llvm.org/use/remote.html).

## Prerequisites

We assume the following setup:

1. [Android NDK is installed](https://developer.android.com/ndk/downloads) and
   the `ANDROID_NDK` environment variable is set to the installation path.
2. Your Android device connected and configured for
   [`adb`](https://developer.android.com/studio/command-line/adb).
3. The Android binary of interest is already compiled and the command to run it
   (in `adb shell`) is `<your-binary> [program args...]`. This does *not* have
   to be a proper Android app with a manifest, etc.

## Running Manually

1. Push the toolchain files, including `lldb-server`, to your device:

    ```shell
    adb shell "mkdir -p /data/local/tmp/tools"
    adb push "$ANDROID_NDK"/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.6/lib/linux/aarch64/* /data/local/tmp/tools
    ```

    You may need to adjust the clang toolchain version to match the one in your
    NDK. You can find it with
    `find "$ANDROID_NDK/toolchains/llvm/prebuilt" -name lldb-server`.

2. Set up port forwarding. We are going to use port 5039 but you are free to
   pick a different one:

    ```shell
    adb forward tcp:5039 tcp:5039
    ```

3. Start an `lldb-server` in a new interactive adb shell:

    ```shell
    adb shell
    /data/local/tmp/tools/lldb-server platform --listen '*:5039' --server
    ```

4. Launch `lldb`, connect to the server and run the binary:

    ```shell
    lldb -o 'platform select remote-android' \
        -o 'platform connect connect://:5039' \
        -o 'platform shell cd /data/local/tmp'
    target create <your-binary>
    run [program args...]
    ```

    You can either use the system `lldb` or a prebuilt under `"$ANDROID_NDK"/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.6/lib/linux/<your-host-arch>`.

    Explanation: each `-o` (short for `--one-shot`) tells lldb to execute a
    command on startup. You can run those manually in the lldb shell, if you
    prefer. Then, we tell lldb which working directory to use, where to find the
    executable, and what command line arguments to use.
