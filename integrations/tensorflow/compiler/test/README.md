# Running tests manually

```shell
$ bazel test :saved_model_adopt_exports
```

This will capture the output and pass it through FileCheck and report pass/fail,
along with a hopefully informative description of what failed.

# Debugging failures

During development, it can be useful to just see the raw output directly.

To see the raw output of the MLIR import and conversion process:

```shell
$ bazel run :saved_model_adopt_exports -- --disable_filecheck
```

Look for the `RUN_TEST: <test_name>` and `FINISH_TEST: <test_name>` lines to
narrow in on the test that interests you.
