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

"""Configures an LLVM overlay project."""

_OVERLAY_PARENT_PATHS = [
    "llvm",
    "mlir",
    "mlir/test",
]

def _is_absolute(path):
    """Returns `True` if `path` is an absolute path.
    Args:
      path: A path (which is a string).
    Returns:
      `True` if `path` is an absolute path.
    """
    return path.startswith("/") or (len(path) > 2 and path[1] == ":")

def _join_path(a, b):
    return str(a) + "/" + str(b)

def _symlink_src_dir(repository_ctx, from_path, to_path):
    children = repository_ctx.path(from_path).readdir()
    for from_child_path in children:
        if to_path:
            to_child_path = _join_path(to_path, from_child_path.basename)
        else:  # Root
            to_child_path = from_child_path.basename

        # Skip paths that are already in the overlay list.
        # They will have this function called on them to populate.
        if to_child_path in _OVERLAY_PARENT_PATHS:
            continue
        repository_ctx.symlink(from_child_path, to_child_path)

def _llvm_configure_impl(repository_ctx):
    # Compute path that sources are symlinked from.
    src_workspace_path = repository_ctx.path(
        repository_ctx.attr.workspace,
    ).dirname
    src_path = repository_ctx.attr.path
    if not _is_absolute(src_path):
        src_path = _join_path(src_workspace_path, src_path)

    # Compute path (relative to here) where overlay files
    # are symlinked from.
    this_workspace_path = repository_ctx.path(
        repository_ctx.attr.workspace,
    ).dirname
    overlay_path = _join_path(
        this_workspace_path,
        "build_tools/bazel/third_party_import/llvm-project/overlay",
    )

    # Each parent path of an overlay file must have its children manually
    # symlinked. This is because the directory itself must be in the cahche
    # (so that we can modify it without modifying the underlying source
    # path). An alternative to this would be to perform a deep symlink of
    # the entire tree (which would be wasteful).
    for overlay_parent_path in _OVERLAY_PARENT_PATHS:
        src_child_path = _join_path(src_path, overlay_parent_path)
        overlay_child_path = _join_path(overlay_path, overlay_parent_path)

        # Symlink from external src path
        _symlink_src_dir(repository_ctx, src_child_path, overlay_parent_path)

        # Symlink from the overlay path.
        _symlink_src_dir(repository_ctx, overlay_child_path, overlay_parent_path)

    # Then symlink any top-level entries not previously handled.
    # Doing it here means that if we got it wrong, it will fail with a
    # "File Exists" error vs doing the wrong thing.
    _symlink_src_dir(repository_ctx, src_path, "")

    # Build files and overlays.
    repository_ctx.file("BUILD")  # Root build is empty.

llvm_configure = repository_rule(
    implementation = _llvm_configure_impl,
    local = True,
    attrs = {
        "_this_workspace": attr.label(default = Label("//:WORKSPACE")),
        "workspace": attr.label(default = Label("//:WORKSPACE")),
        "path": attr.string(mandatory = True),
    },
)
