#!/bin/bash

OUT1=`iree-compile $1 --iree-hal-target-backends=vmvx --iree-opt-const-expr-hoisting=false --iree-opt-const-eval=false --compile-to=executable-targets | grep "vm.mul"`

OUT=$(($?))
if [[ $OUT -eq 0 ]]; then
    exit 0
else
    exit 1
fi
