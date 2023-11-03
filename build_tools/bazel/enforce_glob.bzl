# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A utility to enforce that a list matches a glob expression.

We use this primarily to enable the error-checking capabilities of globs of test
files in IREE while still allowing our Bazel to CMake conversion to not create
CMake globs (which are discouraged for collecting source files, see
https://cmake.org/cmake/help/latest/command/file.html#glob) and not be dependent
on any information outside of the BUILD file.
"""

def enforce_glob(files, **kwargs):
    """A utility to enforce that a list matches a glob expression.

    Note that the comparison is done in an order-independent fashion.

    Args:
        files: a list that is expected to contain the same files as the
            specified glob expression.
        **kwargs: keyword arguments forwarded to the glob.

    Returns:
        files. The input argument unchanged
    """
    glob_result = native.glob(**kwargs)
    for skip_file in ["CMakeLists.txt"]:
        if skip_file in glob_result:
            glob_result.remove(skip_file)

    # glob returns a sorted list.
    if sorted(files) != glob_result:
        glob_result_dict = {k: None for k in glob_result}
        result_dict = {k: None for k in files}
        missing = [k for k in glob_result if k not in files]
        extra = [k for k in files if k not in glob_result]
        expected_formatted = "\n".join(['"{}",'.format(file) for file in glob_result])
        fail(("Error in enforce_glob." +
              "\nExpected {}." +
              "\nGot {}." +
              "\nMissing {}." +
              "\nExtra {}" +
              "\nPaste this into the first enforce_glob argument:" +
              "\n{}").format(
            glob_result,
            files,
            missing,
            extra,
            expected_formatted,
        ))
    return files
