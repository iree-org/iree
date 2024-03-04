/*
 * Copyright 2020-2021 NVIDIA Corporation.  All rights reserved.
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

#ifndef CUDAVDPAUTYPEDEFS_H
#define CUDAVDPAUTYPEDEFS_H

// Dependent includes for cudavdpau.h
#include <vdpau/vdpau.h>

#include <cudaVDPAU.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in cudaVDPAU.h
 */
#define PFN_cuVDPAUGetDevice  PFN_cuVDPAUGetDevice_v3010
#define PFN_cuVDPAUCtxCreate  PFN_cuVDPAUCtxCreate_v3020
#define PFN_cuGraphicsVDPAURegisterVideoSurface  PFN_cuGraphicsVDPAURegisterVideoSurface_v3010
#define PFN_cuGraphicsVDPAURegisterOutputSurface  PFN_cuGraphicsVDPAURegisterOutputSurface_v3010


/**
 * Type definitions for functions defined in cudaVDPAU.h
 */
typedef CUresult (CUDAAPI *PFN_cuVDPAUGetDevice_v3010)(CUdevice_v1 *pDevice, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
typedef CUresult (CUDAAPI *PFN_cuVDPAUCtxCreate_v3020)(CUcontext *pCtx, unsigned int flags, CUdevice_v1 device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
typedef CUresult (CUDAAPI *PFN_cuGraphicsVDPAURegisterVideoSurface_v3010)(CUgraphicsResource *pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuGraphicsVDPAURegisterOutputSurface_v3010)(CUgraphicsResource *pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags);

/*
 * Type definitions for older versioned functions in cudaVDPAU.h
 */
#if defined(__CUDA_API_VERSION_INTERNAL)
typedef CUresult (CUDAAPI *PFN_cuVDPAUCtxCreate_v3010)(CUcontext *pCtx, unsigned int flags, CUdevice_v1 device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
