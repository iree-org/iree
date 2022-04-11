# Dynamic Images Test

This tests runs inference on several images of varying size, ensuring the output
is as expected and timing various parts of the pipeline (e.g. pre-processing,
resizing inputs, inference, post-processing).

Results are saved as a png file in /tmp (x86) or /data/local/tmp (aarch64).

To run on x86:

```shell
bazel build -c opt cpp_projects/dynamic_images:dynamic_images_test
bazel-bin/cpp_projects/dynamic_images/dynamic_images_test
```

To run on aarch64:

```shell
adb push cpp_projects/dynamic_images/test_data /data/local/tmp

bazel build -c opt --config android_arm64 cpp_projects/dynamic_images:dynamic_images_test

adb push bazel-bin/cpp_projects/dynamic_images/dynamic_images_test /data/local/tmp
adb shell chmod +x /data/local/tmp/dynamic_images_test

adb shell /data/local/tmp/dynamic_images_test
```
