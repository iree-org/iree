# Copyright 2020 Google LLC
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

"""Creates a test from the binary output of another rule.

The rule instantiation can pass additional arguments to the binary and provide
it with additional data files (as well as the standard bazel test classification
attributes). This allows compiling the binary once and not recompiling or
relinking it for each test rule. It also avoids a wrapper shell script, which
adds unnecessary shell dependencies and confuses some tooling about the type of
the binary.

Example usage:

run_binary_test(
    name = "my_test",
    args = ["--input_file=$(location :data_file)"],
    data = [":data_file"],
    test_binary = ":some_cc_binary",
)
"""

def _run_binary_test_impl(ctx):
    ctx.actions.run_shell(
        inputs = [ctx.file.test_binary],
        outputs = [ctx.outputs.executable],
        command = "cp $1 $2",
        arguments = [ctx.file.test_binary.path, ctx.outputs.executable.path],
        mnemonic = "CopyExecutable",
    )

    data_runfiles = ctx.runfiles(files = ctx.files.data)

    binary_runfiles = ctx.attr.test_binary[DefaultInfo].default_runfiles

    return [DefaultInfo(
        executable = ctx.outputs.executable,
        runfiles = data_runfiles.merge(binary_runfiles),
    )]

run_binary_test = rule(
    _run_binary_test_impl,
    attrs = {
        "test_binary": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "data": attr.label_list(allow_empty = True, allow_files = True),
    },
    test = True,
)
