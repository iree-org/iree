# IREE TFLite Android Native Bindings

## Building The Library

Process for building the AAR library:

1. Start AndroidStudio. Select _Open File or Project_ then choose `runtime/bindings/tflite/java/gragle.build`
2. AndroidStudio should sync the project and setup gradlew uner `runtime/bindings/tflite/java`
3. Make the project using AndroidStudio or run the build directly in terminal:
```shell
./gradlew build
```

This produces two libraries under `runtime/bindings/tflite/java/build/outputs/aar`:
* `iree-tflite-bindings-debug.aar`
* `iree-tflite-bindings-release.aar`

### AAR Contents

![IREE AAR contents](https://user-images.githubusercontent.com/1041731/121963388-e7680900-cd1e-11eb-89d3-4dee40a42eba.png)

## Using the Library

Include either library in another gradle project for IREE TFLite binding support. See "[Adding dependencies with the Project Structure Dialog](https://developer.android.com/studio/projects/android-library#psd-add-dependencies)" for use in AndroidStudio.
