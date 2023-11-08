# Android App to Run an IREE Bytecode Module

This directory contains configuration to create an Android application that
executes a single IREE module as a native activity.

Note that this app is **purely** for benchmarking/profiling IREE itself.
This is not the expected integration path for a real Android application,
for which we expect to provide proper Java API and build support.

## Native Activity

The app uses Android [`NativeActivity`][native-activity] to bridge IREE core
libraries together with the Android system. Native activity allows one to
implement an Android Activity purely in C/C++. There are
[tutorials][native-activity-tutorial] and [examples][native-activity-example]
one can follow to learn about Native Activity.

## Android Studio

This app does not contain Gradle configurations. The reason is that we need
to package both IREE native libraries and a specific VM module invocation
into the app. The procedure cannot be automated much by Android Studio; rather
one might need to copy files from different places, rename them, and wire up
the build. It's inconvenient. For a developer tool we would like to avoid
such friction and thus improve velocity. So a script,
[`build_apk.sh`](./build_apk.sh), is provided to automate the process.

But the script itself requires an Android SDK/NDK installation structure that
matches Android Studio. So it's easier to just install Android Studio to
manage Android SDK/NDK. The script will use proper tools in Android SDK/NDK
to build and package the final APK file.

## Build the APK

In general, we need to

1. Build the `iree_run_module_app` shared library following normal C++ build
   process.
1. Package the shared library together with an IREE VM FlatBuffer, its entry
   function, input buffers, and the HAL driver into an Android app, following
   a certain directory hierarchy. Specifically,
   1. Generate `AndroidManifest.xml` from the
      [template](./AndroidManifest.xml.template) by providing the proper target
      Android API level.
   1. Copy the VM FlatBuffer as `assets/module.vmfb`, write the entry function
      input buffers, and HAL driver into `assets/entry_function.txt`,
      `assets/inputs.txt`, and `assets/device.txt`, respectively.
   1. Copy the shared library under `lib/<android-abi>/`.
   1. Compile resources under [`res/`](./res) directory into an Android DEX
      file.
   1. Package all of the above into an APK file.
   1. Properly align and sign the APK file.

## Run on Android

When started on Android, the app will read the contents in `assets` to get the
VM FlatBuffer and invocation information and run it.

[native-activity]: https://developer.android.com/reference/android/app/NativeActivity
[native-activity-example]: https://github.com/android/ndk-samples/tree/main/native-activity
[native-activity-tutorial]: https://medium.com/androiddevelopers/getting-started-with-c-and-android-native-activities-2213b402ffff
