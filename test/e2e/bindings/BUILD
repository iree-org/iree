package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_test(
    name = "conv_test",
    srcs = ["conv_test.py"],
    python_version = "PY3",
    deps = [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "//bindings/python/pyiree",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "@absl_py//absl/testing:absltest",
    ],
)
