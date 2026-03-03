#!/bin/zsh
# compile_mips.zsh
#
# Compiles a simple torch.aten.mm through the MIPS custom matmul path:
#   torch.aten.mm
#     -> mips.matmul          (ConvertTorchToMIPSPass)
#     -> flow.dispatch(...)   (IREE dispatch formation)
#     -> func.call @my_matmul_kernel  (bufferize via BufferizableOpInterface)
#     -> LLVM / vmfb          (iree-compile LLVMCPU backend)
#
# Output: /tmp/mm_mips.vmfb
# IR dump: /tmp/mm_mips_ir_dump.mlir  (--mlir-print-ir-after-all)

set -e  # exit on first error

BUILD=/Users/gauravshukla/MLIR_Work/mips/iree-build
IREE_OPT=$BUILD/tools/iree-opt
IREE_COMPILE=$BUILD/tools/iree-compile

# ── Input: 4x4 f32 matrix multiply ────────────────────────────────────────────
cat > /tmp/mm_torch.mlir << 'EOF'
module {
  func.func @mm(%A: !torch.vtensor<[4,4],f32>,
                %B: !torch.vtensor<[4,4],f32>)
      -> !torch.vtensor<[4,4],f32> {
    %0 = torch.aten.mm %A, %B
        : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>
        -> !torch.vtensor<[4,4],f32>
    return %0 : !torch.vtensor<[4,4],f32>
  }
}
EOF

# ── Step 1: Verify torch.aten.mm → mips.matmul ────────────────────────────────
echo "==> Step 1: verifying torch.aten.mm → mips.matmul"
$IREE_OPT \
  --pass-pipeline="builtin.module(func.func(torch-iree-to-mips-matmul))" \
  /tmp/mm_torch.mlir \
  | grep -q "mips.matmul" && echo "    [OK] mips.matmul found in IR"

# ── Step 2: Full torch → IREE input IR (with MIPS path enabled) ───────────────
echo "==> Step 2: torch → IREE input IR  (use-mips-matmul=true)"
$IREE_OPT \
  --pass-pipeline="builtin.module(torch-to-iree{use-mips-matmul=true})" \
  /tmp/mm_torch.mlir \
  -o /tmp/mm_iree.mlir

# ── Step 3: IREE input IR → vmfb (dispatch + bufferize + LLVM) ───────────────
IR_DUMP=/tmp/mm_mips_ir_dump.mlir
echo "==> Step 3: IREE input IR → vmfb  (IR dump → $IR_DUMP)"
$IREE_COMPILE \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-link-embedded=false \
  --mlir-print-ir-after-all \
  /tmp/mm_iree.mlir \
  -o /tmp/mm_mips.vmfb \
  2>"$IR_DUMP"

echo ""
echo "==> Compiled successfully: /tmp/mm_mips.vmfb"
echo "    IR dump written to:    $IR_DUMP"
echo "    Run with:  ./run_mips.zsh"
