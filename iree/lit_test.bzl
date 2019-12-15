# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bazel macros for running lit tests."""

def iree_setup_lit_package(data):
    """Should be called once per test package that contains globbed lit tests.

    Args:
      data: Additional, project specific data deps to add.
    """

    # Bundle together all of the test utilities that are used by tests.
    native.filegroup(
        name = "lit_test_utilities",
        testonly = True,
        data = data + [
            "//iree/tools:IreeFileCheck",
        ],
    )

def iree_glob_lit_tests(
        data = [":lit_test_utilities"],
        driver = "//iree/tools:run_lit.sh",
        test_file_exts = ["mlir"]):
    """Globs lit test files into tests for a package.

    For most packages, the defaults suffice. Packages that include this must
    also include a call to iree_setup_lit_package().

    Args:
      data: Data files to include/build.
      driver: Test driver.
      test_file_exts: File extensions to glob.
    """
    for test_file_ext in test_file_exts:
        test_files = native.glob([
            "*.%s" % (test_file_ext,),
            "**/*.%s" % (test_file_ext,),
        ])
        for test_file in test_files:
            test_file_location = "$(location %s)" % (test_file,)
            native.sh_test(
                name = "%s.test" % (test_file,),
                size = "small",
                srcs = [driver],
                data = data + [test_file],
                args = [test_file_location],
            )
