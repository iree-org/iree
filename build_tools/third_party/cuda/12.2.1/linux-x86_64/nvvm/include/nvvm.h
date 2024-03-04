//
// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2014-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// NVIDIA_COPYRIGHT_END
//

#ifndef NVVM_H
#define NVVM_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>


/*****************************//**
 *
 * \defgroup error Error Handling
 *
 ********************************/


/**
 * \ingroup error
 * \brief   NVVM API call result code.
 */
typedef enum {
  NVVM_SUCCESS = 0,
  NVVM_ERROR_OUT_OF_MEMORY = 1,
  NVVM_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVVM_ERROR_IR_VERSION_MISMATCH = 3,
  NVVM_ERROR_INVALID_INPUT = 4,
  NVVM_ERROR_INVALID_PROGRAM = 5,
  NVVM_ERROR_INVALID_IR = 6,
  NVVM_ERROR_INVALID_OPTION = 7,
  NVVM_ERROR_NO_MODULE_IN_PROGRAM = 8,
  NVVM_ERROR_COMPILATION = 9
} nvvmResult;


/**
 * \ingroup error
 * \brief   Get the message string for the given #nvvmResult code.
 *
 * \param   [in] result NVVM API result code.
 * \return  Message string for the given #nvvmResult code.
 */
const char *nvvmGetErrorString(nvvmResult result);


/****************************************//**
 *
 * \defgroup query General Information Query
 *
 *******************************************/


/**
 * \ingroup query
 * \brief   Get the NVVM version.
 *
 * \param   [out] major NVVM major version number.
 * \param   [out] minor NVVM minor version number.
 * \return 
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *
 */
nvvmResult nvvmVersion(int *major, int *minor);


/**
 * \ingroup query
 * \brief   Get the NVVM IR version.
 *
 * \param   [out] majorIR  NVVM IR major version number.
 * \param   [out] minorIR  NVVM IR minor version number.
 * \param   [out] majorDbg NVVM IR debug metadata major version number.
 * \param   [out] minorDbg NVVM IR debug metadata minor version number.
 * \return 
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *
 */
nvvmResult nvvmIRVersion(int *majorIR, int *minorIR, int *majorDbg, int *minorDbg);


/********************************//**
 *
 * \defgroup compilation Compilation
 *
 ***********************************/

/**
 * \ingroup compilation
 * \brief   NVVM Program.
 *
 * An opaque handle for a program.
 */
typedef struct _nvvmProgram *nvvmProgram;

/**
 * \ingroup compilation
 * \brief   Create a program, and set the value of its handle to \p *prog.
 *
 * \param   [in] prog NVVM program. 
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     nvvmDestroyProgram()
 */
nvvmResult nvvmCreateProgram(nvvmProgram *prog);


/**
 * \ingroup compilation
 * \brief   Destroy a program.
 *
 * \param    [in] prog NVVM program. 
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     nvvmCreateProgram()
 */
nvvmResult nvvmDestroyProgram(nvvmProgram *prog);


/**
 * \ingroup compilation
 * \brief   Add a module level NVVM IR to a program. 
 *
 * The \p buffer should contain an NVVM IR module.
 * The module should have NVVM IR either in the LLVM 7.0.1 bitcode
 * representation or in the LLVM 7.0.1 text representation. Support for reading
 * the text representation of NVVM IR is deprecated and may be removed in a
 * later version.
 *
 * \param   [in] prog   NVVM program.
 * \param   [in] buffer NVVM IR module in the bitcode or text
 *                      representation.
 * \param   [in] size   Size of the NVVM IR module.
 * \param   [in] name   Name of the NVVM IR module.
 *                      If NULL, "<unnamed>" is used as the name.
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_INPUT \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 */
nvvmResult nvvmAddModuleToProgram(nvvmProgram prog, const char *buffer, size_t size, const char *name);

/**
 * \ingroup compilation
 * \brief   Add a module level NVVM IR to a program. 
 *
 * The \p buffer should contain an NVVM IR module. The module should have NVVM
 * IR in the LLVM 7.0.1 bitcode representation.
 *
 * A module added using this API is lazily loaded - the only symbols loaded
 * are those that are required by module(s) loaded using
 * nvvmAddModuleToProgram. It is an error for a program to have
 * all modules loaded using this API. Compiler may also optimize entities
 * in this module by making them internal to the linked NVVM IR module,
 * making them eligible for other optimizations. Due to these
 * optimizations, this API to load a module is more efficient and should
 * be used where possible.
 * 
 * \param   [in] prog   NVVM program.
 * \param   [in] buffer NVVM IR module in the bitcode representation.
 * \param   [in] size   Size of the NVVM IR module.
 * \param   [in] name   Name of the NVVM IR module.
 *                      If NULL, "<unnamed>" is used as the name.
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_INPUT \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 */
nvvmResult nvvmLazyAddModuleToProgram(nvvmProgram prog, const char *buffer, size_t size, const char *name);

/**
 * \ingroup compilation
 * \brief   Compile the NVVM program.
 *
 * The NVVM IR modules in the program will be linked at the IR level.
 * The linked IR program is compiled to PTX.
 *
 * The target datalayout in the linked IR program is used to
 * determine the address size (32bit vs 64bit).
 *
 * The valid compiler options are:
 *
 *   - -g (enable generation of full debugging information).
 *        Full debug support is only valid with '-opt=0'. Debug support
 *        requires the input module to utilize NVVM IR Debug Metadata.
 *        Line number (line info) only generation is also enabled via NVVM IR
 *        Debug Metadata, there is no specific libNVVM API flag for that case.
 *   - -opt=
 *     - 0 (disable optimizations)
 *     - 3 (default, enable optimizations)
 *   - -arch=
 *     - compute_50
 *     - compute_52 (default)
 *     - compute_53
 *     - compute_60
 *     - compute_61
 *     - compute_62
 *     - compute_70
 *     - compute_72
 *     - compute_75
 *     - compute_80
 *     - compute_87
 *     - compute_89
 *     - compute_90
 *   - -ftz=
 *     - 0 (default, preserve denormal values, when performing
 *          single-precision floating-point operations)
 *     - 1 (flush denormal values to zero, when performing
 *          single-precision floating-point operations)
 *   - -prec-sqrt=
 *     - 0 (use a faster approximation for single-precision
 *          floating-point square root)
 *     - 1 (default, use IEEE round-to-nearest mode for
 *          single-precision floating-point square root)
 *   - -prec-div=
 *     - 0 (use a faster approximation for single-precision
 *          floating-point division and reciprocals)
 *     - 1 (default, use IEEE round-to-nearest mode for
 *          single-precision floating-point division and reciprocals)
 *   - -fma=
 *     - 0 (disable FMA contraction)
 *     - 1 (default, enable FMA contraction)
 *
 * \param   [in] prog       NVVM program.
 * \param   [in] numOptions Number of compiler \p options passed.
 * \param   [in] options    Compiler options in the form of C string array.
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::nvvmResult NVVM_ERROR_IR_VERSION_MISMATCH \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_OPTION \endlink
 *   - \link ::nvvmResult NVVM_ERROR_NO_MODULE_IN_PROGRAM \endlink
 *   - \link ::nvvmResult NVVM_ERROR_COMPILATION \endlink
 */
nvvmResult nvvmCompileProgram(nvvmProgram prog, int numOptions, const char **options);   

/**
 * \ingroup compilation
 * \brief   Verify the NVVM program.
 *
 * The valid compiler options are:
 *
 * Same as for nvvmCompileProgram().
 *
 * \param   [in] prog       NVVM program.
 * \param   [in] numOptions Number of compiler \p options passed.
 * \param   [in] options    Compiler options in the form of C string array.
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::nvvmResult NVVM_ERROR_IR_VERSION_MISMATCH \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_IR \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_OPTION \endlink
 *   - \link ::nvvmResult NVVM_ERROR_NO_MODULE_IN_PROGRAM \endlink
 *
 * \see     nvvmCompileProgram()
 */
nvvmResult nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char **options);

/**
 * \ingroup compilation
 * \brief   Get the size of the compiled result.
 *
 * \param   [in]  prog          NVVM program.
 * \param   [out] bufferSizeRet Size of the compiled result (including the
 *                              trailing NULL).
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 */
nvvmResult nvvmGetCompiledResultSize(nvvmProgram prog, size_t *bufferSizeRet);


/**
 * \ingroup compilation
 * \brief   Get the compiled result.
 *
 * The result is stored in the memory pointed to by \p buffer.
 *
 * \param   [in]  prog   NVVM program.
 * \param   [out] buffer Compiled result.
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 */
nvvmResult nvvmGetCompiledResult(nvvmProgram prog, char *buffer);


/**
 * \ingroup compilation
 * \brief   Get the Size of Compiler/Verifier Message.
 *
 * The size of the message string (including the trailing NULL) is stored into
 * \p bufferSizeRet when the return value is NVVM_SUCCESS.
 *   
 * \param   [in]  prog          NVVM program.
 * \param   [out] bufferSizeRet Size of the compilation/verification log
                                (including the trailing NULL).
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 */
nvvmResult nvvmGetProgramLogSize(nvvmProgram prog, size_t *bufferSizeRet);


/**
 * \ingroup compilation
 * \brief   Get the Compiler/Verifier Message.
 *
 * The NULL terminated message string is stored in the memory pointed to by
 * \p buffer when the return value is NVVM_SUCCESS.
 *   
 * \param   [in]  prog   NVVM program.
 * \param   [out] buffer Compilation/Verification log.
 * \return
 *   - \link ::nvvmResult NVVM_SUCCESS \endlink
 *   - \link ::nvvmResult NVVM_ERROR_INVALID_PROGRAM \endlink
 */
nvvmResult nvvmGetProgramLog(nvvmProgram prog, char *buffer);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NVVM_H */
