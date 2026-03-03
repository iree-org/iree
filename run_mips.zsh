#!/bin/zsh
# run_mips.zsh
#
# Runs the vmfb produced by compile_mips.zsh.
#
# The vmfb's dispatch executable calls @my_matmul_kernel through the IREE
# import mechanism (not direct dynamic linking).  The kernel is provided by
# libmy_matmul_kernel.dylib which implements the IREE executable plugin API
# (exports iree_hal_executable_plugin_query).
#
# Test: A * I = A  (multiply by 4x4 identity → expect the same matrix back)

BUILD=/Users/gauravshukla/MLIR_Work/mips/iree-build
KERNEL_LIB=$BUILD/runtime/src/iree/builtins/mips/libmy_matmul_kernel.dylib
IREE_RUN=$BUILD/tools/iree-run-module
VMFB=/tmp/mm_mips.vmfb

if [[ ! -f $VMFB ]]; then
  echo "ERROR: $VMFB not found. Run compile_mips.zsh first."
  exit 1
fi

if [[ ! -f $KERNEL_LIB ]]; then
  echo "ERROR: $KERNEL_LIB not found. Build my_matmul_kernel target first."
  exit 1
fi

# A = 1..16 (row-major 4x4), B = 4x4 identity
A="4x4xf32=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
B="4x4xf32=1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1"

echo "==> Running mm(A, I) via MIPS kernel"
echo "    Kernel plugin : $KERNEL_LIB"
echo "    Expected      : A * I = A  (rows: [1 2 3 4], [5 6 7 8], ...)"
echo ""

$IREE_RUN \
  --executable_plugin=$KERNEL_LIB \
  --module=$VMFB \
  --function=mm \
  --input="$A" \
  --input="$B"
