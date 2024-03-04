
TOP              = $(_HERE_)/..

NVVMIR_LIBRARY_DIR = $(TOP)/$(_NVVM_BRANCH_)/libdevice

LD_LIBRARY_PATH += $(TOP)/lib:
PATH            += $(TOP)/$(_NVVM_BRANCH_)/bin:$(_HERE_):

INCLUDES        +=  "-I$(TOP)/$(_TARGET_DIR_)/include" $(_SPACE_)

LIBRARIES        =+ $(_SPACE_) "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)/stubs" "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)"

CUDAFE_FLAGS    +=
PTXAS_FLAGS     +=
