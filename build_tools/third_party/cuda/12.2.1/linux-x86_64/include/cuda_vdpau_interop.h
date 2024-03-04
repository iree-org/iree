/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__CUDA_VDPAU_INTEROP_H__)
#define __CUDA_VDPAU_INTEROP_H__

#include "cuda_runtime_api.h"

#include <vdpau/vdpau.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \addtogroup CUDART_VDPAU VDPAU Interoperability
 * This section describes the VDPAU interoperability functions of the CUDA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Gets the CUDA device associated with a VdpDevice.
 *
 * Returns the CUDA device associated with a VdpDevice, if applicable.
 *
 * \param device - Returns the device associated with vdpDevice, or -1 if
 * the device associated with vdpDevice is not a compute device.
 * \param vdpDevice - A VdpDevice handle
 * \param vdpGetProcAddress - VDPAU's VdpGetProcAddress function pointer
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa
 * ::cudaVDPAUSetVDPAUDevice,
 * ::cuVDPAUGetDevice
 */
extern __host__ cudaError_t CUDARTAPI cudaVDPAUGetDevice(int *device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);

/**
 * \brief Sets a CUDA device to use VDPAU interoperability
 *
 * Records \p vdpDevice as the VdpDevice for VDPAU interoperability 
 * with the CUDA device \p device and sets \p device as the current 
 * device for the calling host thread.
 *
 * This function will immediately initialize the primary context on 
 * \p device if needed.
 *
 * If \p device has already been initialized then this call will fail 
 * with the error ::cudaErrorSetOnActiveProcess.  In this case it is 
 * necessary to reset \p device using ::cudaDeviceReset() before 
 * VDPAU interoperability on \p device may be enabled.
 *
 * \param device - Device to use for VDPAU interoperability
 * \param vdpDevice - The VdpDevice to interoperate with
 * \param vdpGetProcAddress - VDPAU's VdpGetProcAddress function pointer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorSetOnActiveProcess
 * \notefnerr
 *
 * \sa ::cudaGraphicsVDPAURegisterVideoSurface,
 * ::cudaGraphicsVDPAURegisterOutputSurface,
 * ::cudaDeviceReset
 */
extern __host__ cudaError_t CUDARTAPI cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);

/**
 * \brief Register a VdpVideoSurface object
 *
 * Registers the VdpVideoSurface specified by \p vdpSurface for access by CUDA.
 * A handle to the registered object is returned as \p resource.
 * The surface's intended usage is specified using \p flags, as follows:
 *
 * - ::cudaGraphicsMapFlagsNone: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::cudaGraphicsMapFlagsReadOnly: Specifies that CUDA
 *   will not write to this resource.
 * - ::cudaGraphicsMapFlagsWriteDiscard: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * \param resource - Pointer to the returned object handle
 * \param vdpSurface - VDPAU object to be registered
 * \param flags - Map flags
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::cudaVDPAUSetVDPAUDevice,
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsVDPAURegisterVideoSurface
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsVDPAURegisterVideoSurface(struct cudaGraphicsResource **resource, VdpVideoSurface vdpSurface, unsigned int flags);

/**
 * \brief Register a VdpOutputSurface object
 *
 * Registers the VdpOutputSurface specified by \p vdpSurface for access by CUDA.
 * A handle to the registered object is returned as \p resource.
 * The surface's intended usage is specified using \p flags, as follows:
 *
 * - ::cudaGraphicsMapFlagsNone: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::cudaGraphicsMapFlagsReadOnly: Specifies that CUDA
 *   will not write to this resource.
 * - ::cudaGraphicsMapFlagsWriteDiscard: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * \param resource - Pointer to the returned object handle
 * \param vdpSurface - VDPAU object to be registered
 * \param flags - Map flags
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::cudaVDPAUSetVDPAUDevice,
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsVDPAURegisterOutputSurface
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsVDPAURegisterOutputSurface(struct cudaGraphicsResource **resource, VdpOutputSurface vdpSurface, unsigned int flags);

/** @} */ /* END CUDART_VDPAU */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* __CUDA_VDPAU_INTEROP_H__ */

