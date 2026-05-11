#!/bin/bash
echo Running: ../build/tools/iree-compile --iree-hal-target-device=hip --iree-rocm-target=gfx1201 --iree-rocm-use-spirv --iree-hal-dump-executable-intermediates-to=intermediates matmul_f16.mlir -o spvbin.vmfb
../build/tools/iree-compile --iree-hal-target-device=hip --iree-rocm-target=gfx1201 --iree-rocm-use-spirv --iree-hal-dump-executable-intermediates-to=/tmp/jit matmul_f16.mlir -o spvbin.vmfb

echo Extract intermediate SPIRV disassembly
spirv-dis /tmp/jit/*.spv > disassembly/f16jitspirv.asm

echo Obtain jit ISA
AMD_COMGR_SAVE_TEMPS=1 ../build/tools/iree-run-module --module=spvbin.vmfb --device=hip \
  --function=main --input=...
llvm-objdump -d /tmp/comgr-*/output/a.so > disassembly/f16jitrocmasm.asm

echo Now Running: ../build/tools/iree-compile --iree-hal-target-device=hip --iree-rocm-target=gfx1201 --iree-hal-dump-executable-intermediates-to=intermediates matmul_f16.mlir -o aotbin.vmfb
../build/tools/iree-compile --iree-hal-target-device=hip --iree-rocm-target=gfx1201 --iree-hal-dump-executable-intermediates-to=/tmp/aot matmul_f16.mlir -o aotbin.vmfb

echo Obtain AOT ISA
cat /tmp/aot/*.rocmasm > disassembly/f16aotrocmasm.asm

echo Test run aot
../build/tools/iree-run-module --module=aotbin.vmfb --device=hip \
  --function=main --input=...

