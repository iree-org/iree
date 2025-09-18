FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:405945a40deaff9db90b9839c0f41d4cba4a383c1a7459b28627047bf6302a26

# Install system dependencies and ROCm in the image
RUN apt-get update && \
    apt-get install -y \
        gh \
        gfortran \
        build-essential \
        binutils \
        python3-pip \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Install ROCm packages
RUN python3 -m pip install \
    --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ \
    rocm[libraries,devel]

# Set ROCm environment variables
RUN export ROCM_PATH=$(python3 -m rocm_sdk path --root) && \
    export LD_LIBRARY_PATH="$ROCM_PATH/lib":$LD_LIBRARY_PATH && \
    echo "export ROCM_PATH=$ROCM_PATH" >> $GITHUB_ENV && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $GITHUB_ENV

# Verify ROCm installation
RUN python3 -c "import rocm_sdk; print('ROCm SDK installed successfully')"
