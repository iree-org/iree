import os
import sys

import lit.formats
import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)

config.name = "RISC-V tests"
config.test_format = lit.formats.ShTest(True)

config.suffixes = [".run"]

config.environment["BUILD_RISCV_DIR"] = os.getenv("BUILD_RISCV_DIR")

config.environment["TEST_CMD"] = (
    "%s -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0"
    " -L %s/sysroot " %
    (os.getenv("QEMU_RV64_BIN"), os.getenv("RISCV_TOOLCHAIN_ROOT")))

config.environment["TEST_MODULE_CMD"] = (
    "%s %s/tools/iree-run-module --driver=local-task" %
    (config.environment["TEST_CMD"], os.getenv("BUILD_RISCV_DIR")))

config.test_exec_root = os.getenv("BUILD_RISCV_DIR") + \
    "/tests"
