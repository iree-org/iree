if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi
~/nod/iree-build-notrace/install/bin/iree-compile ukernel_example.mlir --iree-rocm-target-chip=$1 --iree-input-type=torch --iree-hal-target-backends=rocm --iree-rocm-enable-ukernels=argmax --iree-rocm-link-bc=true --verify=true -o ukernel_argmax.vmfb --mlir-print-ir-after-all 2> cool.mlir
# Expects 237 this is hardcoded in value_generator.py
python value_generator.py
echo "If it works correctly, result should be 237, this is hardcoded from the py script."
~/nod/iree-build-notrace/install/bin/iree-run-module --module=ukernel_argmax.vmfb --device=rocm --function=forward --input=@input0.npy --device_allocator=caching
echo "Now testing for fp16, if it works correctly, result should still be 237."
~/nod/iree-build-notrace/install/bin/iree-run-module --module=ukernel_argmax.vmfb --device=rocm --function=forward_f16 --input=@input0_f16.npy --device_allocator=caching
# Clean up.
rm ukernel_argmax.vmfb argmax.bc input0.npy input0_f16.npy