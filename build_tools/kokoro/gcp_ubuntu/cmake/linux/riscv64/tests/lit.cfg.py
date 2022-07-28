import os
import sys
import tempfile

import lit.formats

config.name = "RISC-V tests"
config.test_format = lit.formats.ShTest(execute_external=True)

config.suffixes = [".run"]

BUILD_RISCV_DIR = os.path.abspath(os.environ["BUILD_RISCV_DIR"])

config.environment["BUILD_RISCV_DIR"] = BUILD_RISCV_DIR

test_cmd = [
    os.environ["QEMU_RV64_BIN"],
    "-cpu",
    "rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0",
    "-L",
    os.path.join(os.environ["RISCV_TOOLCHAIN_ROOT"], "sysroot"),
]

config.environment["TEST_CMD"] = " ".join(test_cmd)

test_module_cmd = test_cmd + [
    os.path.join(BUILD_RISCV_DIR, "tools/iree-run-module"),
    "--device=local-task"
]

config.environment["TEST_MODULE_CMD"] = " ".join(test_module_cmd)

# Use the most preferred temp directory.
config.test_exec_root = (os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or
                         os.environ.get("TEST_TMPDIR") or
                         os.path.join(tempfile.gettempdir(), "lit"))
