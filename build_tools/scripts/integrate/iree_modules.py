# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class ModuleInfo:

    def __init__(self, *, name: str, path: str, branch_pin_file: str,
                 default_repository_url: str, fork_repository_push: str,
                 fork_repository_pull: str, branch_prefix: str):
        self.name = name
        self.path = path
        self.branch_pin_file = branch_pin_file
        self.default_repository_url = default_repository_url
        self.fork_repository_push = fork_repository_push
        self.fork_repository_pull = fork_repository_pull
        self.branch_prefix = branch_prefix


MODULE_INFOS = {
    "llvm-project":
    ModuleInfo(
        name="llvm-project",
        path="third_party/llvm-project",
        branch_pin_file="third_party/llvm-project.branch-pin",
        default_repository_url="https://github.com/iree-org/iree-llvm-fork.git",
        fork_repository_push="git@github.com:iree-org/iree-llvm-fork.git",
        fork_repository_pull="https://github.com/iree-org/iree-llvm-fork.git",
        branch_prefix="patched-llvm-project-",
    ),
    "mlir-hlo":
    ModuleInfo(
        name="mlir-hlo",
        path="third_party/mlir-hlo",
        branch_pin_file="third_party/mlir-hlo.branch-pin",
        default_repository_url="https://github.com/iree-org/iree-mhlo-fork.git",
        fork_repository_push="git@github.com:iree-org/iree-mhlo-fork.git",
        fork_repository_pull="https://github.com/iree-org/iree-mhlo-fork.git",
        branch_prefix="patched-mlir-hlo-",
    )
}
