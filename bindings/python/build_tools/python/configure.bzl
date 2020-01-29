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

"""Automatically detects and configures for native python building."""

_ENV_PYTHON_BIN = "PYTHON_BIN"

def _python_configure_impl(repository_ctx):
    # Symlink build_defs.bzl into the new repository.
    repository_ctx.symlink(repository_ctx.attr._build_defs, "build_defs.bzl")

    python_bin = _get_python_bin(repository_ctx)
    generate_script = repository_ctx.path(repository_ctx.attr._generate_script)

    # print("Generator: %s" % generate_script)
    exec_result = repository_ctx.execute([python_bin, generate_script])
    if exec_result.return_code != 0:
        fail(("Failed to execute python configure script: %s %s " +
              "(stderr follows)\n%s") % (
            python_bin,
            generate_script,
            exec_result.stderr,
        ))
    build_contents = exec_result.stdout

    # print("Build contents: %s" % build_contents)
    # print("Stderr: %s" % (exec_result.stderr,))
    repository_ctx.file("BUILD", build_contents)

    # Now parse the build_contents for directives on how to setup the repo.
    build_lines = build_contents.splitlines()
    SYMLINK_PREFIX = "# SYMLINK: "
    for build_line in build_lines:
        if build_line.startswith(SYMLINK_PREFIX):
            symlink_from, symlink_to = build_line[len(SYMLINK_PREFIX):].split("\t")

            # print('SYMLINK:', symlink_from, symlink_to)
            repository_ctx.symlink(symlink_from, symlink_to)

python_configure = repository_rule(
    implementation = _python_configure_impl,
    environ = [
        _ENV_PYTHON_BIN,
        "PATH",
    ],
    attrs = {
        "_generate_script": attr.label(
            default = Label("//bindings/python/build_tools/python:generate_build.py"),
            allow_single_file = True,
        ),
        "_build_defs": attr.label(
            default = Label("//bindings/python/build_tools/python:build_defs.bzl"),
            allow_single_file = True,
        ),
    },
)

def _get_python_bin(repository_ctx):
    python_bin = repository_ctx.os.environ.get(_ENV_PYTHON_BIN)
    if python_bin != None:
        print("Using python binary from %s = %s" % (
            _ENV_PYTHON_BIN,
            python_bin,
        ))  # buildozer: disable=print

        return python_bin
    python_bin_path = repository_ctx.which("python")
    if python_bin != None:
        print("Using python from system PATH: %s" % (python_bin,))  # buildozer: disable=print
        return str(python_bin)  # buildozer: disable=print

    fail((
        "Unable to find python binary (via %s on PATH %s) " +
        "Note that PATH resolution on Windows is unreliable. Prefer " +
        "explicit configuration."
    ) % (
        _ENV_PYTHON_BIN,
        repository_ctx.os.environ.get("PATH", ""),
    ))
