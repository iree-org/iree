#!/bin/bash

OUT1=`./tools/iree-compile $1 --iree-hal-target-backends=cuda --iree-opt-const-expr-hoisting=false --iree-opt-const-eval=false --compile-to=executable-targets | grep "llvm.intr.vector.reduce.fmax"`

OUT=$(($?))
if [[ $OUT -eq 0 ]]; then
    exit 0
else
    exit 1
fi
