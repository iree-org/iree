# Cloned TFLite C API Headers

These files were captured from the tensorflow repository:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/

Slight modifications were required in order to get them compiling in C and
outside of the tensorflow repository. Features that IREE does not support are
guarded with `IREE_BINDINGS_TFLITE_INCLUDE_UNSUPPORTED_APIS`.

Origin commit: f954b2770d0cfd8244a9cf9d7116fb15bc044118
https://github.com/tensorflow/tensorflow/tree/f954b2770d0cfd8244a9cf9d7116fb15bc044118
