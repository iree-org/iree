# iree-run-module Android App

This directory contains an Android application packing the `iree-run-module`
command-line tool together with a specific IREE VM module invocation.
This is helpful for profiling/debugging using Android tools.

## Native Activity

The App uses Android [`NativeActivity`][native-activity] to bridge IREE core
libraries together with the Android system. Native activity allows one to
implement an Android Activity purely in C/C++. There are
[tutorials][native-activity-tutorial] and [examples][native-activity-example]
one can follow to learn about Native Activity.

## Build the APK

We need to package IREE native libraries and a specific VM module invocation.
IDE is not very convenient to perform such tasks, so a script,
[`build_apk.sh`](./build_apk.sh), is provided to automate the process.

In general, we need to

1. Build the `iree_run_module_app` shared library following normal C++ build
   process.
1. Package the shared library together with an IREE VM FlatBuffer, its entry
   function, input buffers, and the HAL driver into an Android app, following
   a certain directory hierarchy. Specifically,
   1. Generating `AndroidManifest.xml` from the
      [template](./AndroidManifest.xml.template) by providing the proper target
      Android API level.
   1. Copy the VM FlatBuffer as `assets/module.vmfb`, write the entry function
      input buffers, and HAL driver into `assets/entry_function.txt`,
      `assets/inputs.txt`, and `assets/driver.txt`, respectively.
   1. Copy the shared libary under `lib/<android-abi>/`.
   1. Compile resources under [`res/`](./res) directory into an Android DEX
      file.
   1. Package all of the above into an APK file.
   1. Properly align and sign the APK file.

## Run on Android

When started on Android, the app will read the contents in `assets` to get the
VM FlatBuffer and invocation information and run it.

[native-activity]: https://developer.android.com/reference/android/app/NativeActivity
[native-activity-example]: https://github.com/android/ndk-samples/tree/master/native-activity
[native-activity-tutorial]: https://medium.com/androiddevelopers/getting-started-with-c-and-android-native-activities-2213b402ffff
