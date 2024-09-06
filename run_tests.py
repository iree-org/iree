import subprocess
import numpy as np
import torch

def run_python_file(python_file):
    try:
        result = subprocess.run(['python', python_file], check=True)
        print(f"Python file '{python_file}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing '{python_file}': {e}")

def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"Bash command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing bash command '{command}': {e}")

def compare_npy_files(file1, file2):
    try:
        array1 = np.load(file1)
        array2 = np.load(file2)

        print(array1.shape)
        print(array2.shape)

        print(array1[0][0][0])
        print(array2[0][0][0])

        torch.testing.assert_close(array1, array2)

        if np.array_equal(array1, array2):
            print(f"Files '{file1}' and '{file2}' are identical.")
        else:
            print(f"Files '{file1}' and '{file2}' differ.")
    except Exception as e:
        print(f"Error comparing files '{file1}' and '{file2}': {e}")

if __name__ == "__main__":
    python_file = 'test.py'
    bash_command_1 = 'build/tools/iree-compile test_attn.mlir --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx1100 --iree-global-opt-propagate-transposes=true --iree-opt-outer-dim-concat=true --iree-opt-const-eval=false --iree-opt-data-tiling=false --iree-rocm-waves-per-eu=2 --iree-vm-target-truncate-unsupported-floats --iree-codegen-llvmgpu-use-vector-distribution --iree-codegen-gpu-native-math-precision=true --iree-flow-enable-aggressive-fusion -o fused_attn.vmfb'
    bash_command_2 = 'build/tools/iree-run-module --module=fused_attn.vmfb --device=hip --input=@attn_q.npy --input=@attn_k.npy --input=@attn_v.npy --input=@attn_mask.npy --output=@attn_out.npy'
    npy_file_1 = 'attn_out.npy'
    npy_file_2 = 'attn_ref.npy'

    run_python_file(python_file)
    run_bash_command(bash_command_1)
    run_bash_command(bash_command_2)
    compare_npy_files(npy_file_1, npy_file_2)
