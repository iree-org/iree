if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi
~/nod/iree-build-notrace/llvm-project/bin/clang -x hip --offload-arch=$1 --offload-device-only -nogpulib -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xclang -disable-llvm-optzns -O3 -fvisibility=protected -fomit-frame-pointer -Xclang -finclude-default-header -Xclang -fexperimental-strict-floating-point  -Xclang -fdenormal-fp-math=dynamic -emit-llvm -c argmax_ukernel.c -o argmax.bc
~/nod/iree-build-notrace/install/bin/iree-compile ukernel_example.mlir --iree-rocm-target-chip=$1 --iree-input-type=torch --iree-hal-target-backends=rocm --iree-rocm-enable-ukernels=argmax --iree-rocm-link-bc=true --verify=true -o ukernel_argmax.vmfb --iree-link-bitcode=/home/stwinata/nod/iree/samples/custom_dispatch/cuda/argmax_kernels/argmax.bc
# Expects 237 this is hardcoded in value_generator.py
python value_generator.py
echo "If it works correctly, result should be 237, this is hardcoded from the py script."
~/nod/iree-build-notrace/install/bin/iree-benchmark-module --module=ukernel_argmax.vmfb --device=rocm --function=forward --input=@input0.npy --device_allocator=caching
echo "Now testing for fp16, if it works correctly, result should still be 237."
~/nod/iree-build-notrace/install/bin/iree-benchmark-module --module=ukernel_argmax.vmfb --device=rocm --function=forward_f16 --input=@input0_f16.npy --device_allocator=caching
# Clean up.
rm ukernel_argmax.vmfb argmax.bc input0.npy input0_f16.npy