# Copyright 2023 The IREE Authors

set -x
set -e
CLANG="${CLANG:-clang++}"
LLVMAS="${LLVMAS:-llvm-as}"
NVCC="${NVCC:-nvcc}"
IREE_SRC_DIR="$(git rev-parse --show-toplevel)"

SCRIPT_DIR="$(realpath `dirname $0`)"
OUT="${SCRIPT_DIR?}/"
SRC="${SCRIPT_DIR?}/"
# todo(guray) Where is the CUTLASS?
CUTLASS="/usr/local/google/home/gurayozen/work/cutlass/"

function make_arch_bc {
  local SM=$1  
  local SOURCE=$2
  local SOURCE_FILE=$SOURCE.cu
  local FILE_BASENAME="${OUT}/microkernels_${SM}"
  ${CLANG?} \
      -O3 \
      -std=c++17 \
      -Xclang -fcuda-allow-variadic-functions \
      --cuda-gpu-arch=sm_${SM} \
      -D__CUDACC_VER_MAJOR__=11 \
      -D__CUDA_ARCH__=${SM}0 \
      -I.. \
      -I"${CUTLASS}"/include \
      -I"${CUTLASS}"/tools/util/include/ \
      "${SRC}/${SOURCE_FILE}" \
      -S \
      -emit-llvm
  ${LLVMAS?} $SOURCE-cuda-nvptx64-nvidia-cuda-sm_${SM}.ll -o ukernels-cuda-nvptx64-nvidia-cuda-sm_${SM}.bc
}

function make_arch_nvcc_ptx {
  local SM=$1  
  local SOURCE_FILE=$2.cu
  ${NVCC?} \
      -arch sm_${SM} \
      -I.. \
      -I"${CUTLASS}"/include \
      -I"${CUTLASS}"/tools/util/include/ \
      "${SRC}/${SOURCE_FILE}" \
      -dc \
      --ptx   
  #  Following can be added to see cuda codes in the profiler
  # -lineinfo
  
  # Remove these since nvvm also generates
  sed -i 's/.version 7.8//g' $2.ptx
  sed -i 's/.address_size 64//g' $2.ptx
  sed -i 's/.target sm_80//g' $2.ptx 

  mv $2.ptx ukernels-cuda-nvptx64-nvidia-cuda-sm_${SM}.ptx
}

function make_arch_nvcc_lineinfo_ptx {
  local SM=$1  
  local SOURCE_FILE=$2.cu
  ${NVCC?} \
      -arch sm_${SM} \
      -I.. \
      -I"${CUTLASS}"/include \
      -I"${CUTLASS}"/tools/util/include/ \
      "${SRC}/${SOURCE_FILE}" \
      -dc \
      --ptx \
      -lineinfo
  
  # Remove these since nvvm also generates
  sed -i 's/.version 7.8//g' $2.ptx
  sed -i 's/.address_size 64//g' $2.ptx
  sed -i 's/.target sm_80//g' $2.ptx 

  mv $2.ptx ukernels-cuda-debug-nvptx64-nvidia-cuda-sm_${SM}.ptx
}

function generate_generator {
  local SOURCE=$1
  local GENERATOR_FILE=$2.cu
  rm -f $GENERATOR_FILE
  rm -f GenerateKernels
  ${CLANG?} -std=c++17 -O3 $SOURCE -o GenerateKernels
  ./GenerateKernels $GENERATOR_FILE
  rm -f GenerateKernels  
}

function generate_ukernels {
  local GENERATOR_FILE=$1
  local SM=$2

  # Generate Template instantiater
  generate_generator ../uCUDAKernelGenerator.cpp $GENERATOR_FILE
  
  # # Generate microkernels
  make_arch_nvcc_ptx $SM $GENERATOR_FILE
  make_arch_nvcc_lineinfo_ptx $SM $GENERATOR_FILE
  make_arch_bc $SM $GENERATOR_FILE

  # # Remove the temps
  rm -rf $GENERATOR_FILE*
}

generate_ukernels "TemplateInstantiator" 80
