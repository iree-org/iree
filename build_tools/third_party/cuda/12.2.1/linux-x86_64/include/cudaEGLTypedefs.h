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

#ifndef CUDAEGLTYPEDEFS_H
#define CUDAEGLTYPEDEFS_H

#include <cudaEGL.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in cudaEGL.h
 */
#define PFN_cuGraphicsEGLRegisterImage  PFN_cuGraphicsEGLRegisterImage_v7000
#define PFN_cuEGLStreamConsumerConnect  PFN_cuEGLStreamConsumerConnect_v7000
#define PFN_cuEGLStreamConsumerConnectWithFlags  PFN_cuEGLStreamConsumerConnectWithFlags_v8000
#define PFN_cuEGLStreamConsumerDisconnect  PFN_cuEGLStreamConsumerDisconnect_v7000
#define PFN_cuEGLStreamConsumerAcquireFrame  PFN_cuEGLStreamConsumerAcquireFrame_v7000
#define PFN_cuEGLStreamConsumerReleaseFrame  PFN_cuEGLStreamConsumerReleaseFrame_v7000
#define PFN_cuEGLStreamProducerConnect  PFN_cuEGLStreamProducerConnect_v7000
#define PFN_cuEGLStreamProducerDisconnect  PFN_cuEGLStreamProducerDisconnect_v7000
#define PFN_cuEGLStreamProducerPresentFrame  PFN_cuEGLStreamProducerPresentFrame_v7000
#define PFN_cuEGLStreamProducerReturnFrame  PFN_cuEGLStreamProducerReturnFrame_v7000
#define PFN_cuGraphicsResourceGetMappedEglFrame  PFN_cuGraphicsResourceGetMappedEglFrame_v7000
#define PFN_cuEventCreateFromEGLSync  PFN_cuEventCreateFromEGLSync_v9000


/**
 * Type definitions for functions defined in cudaEGL.h
 */
typedef CUresult (CUDAAPI *PFN_cuGraphicsEGLRegisterImage_v7000)(CUgraphicsResource CUDAAPI *pCudaResource, EGLImageKHR image, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamConsumerConnect_v7000)(CUeglStreamConnection CUDAAPI *conn, EGLStreamKHR stream);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamConsumerConnectWithFlags_v8000)(CUeglStreamConnection CUDAAPI *conn, EGLStreamKHR stream, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamConsumerDisconnect_v7000)(CUeglStreamConnection CUDAAPI *conn);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamConsumerAcquireFrame_v7000)(CUeglStreamConnection CUDAAPI *conn, CUgraphicsResource CUDAAPI *pCudaResource, CUstream CUDAAPI *pStream, unsigned int timeout);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamConsumerReleaseFrame_v7000)(CUeglStreamConnection CUDAAPI *conn, CUgraphicsResource pCudaResource, CUstream CUDAAPI *pStream);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamProducerConnect_v7000)(CUeglStreamConnection CUDAAPI *conn, EGLStreamKHR stream, EGLint width, EGLint height);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamProducerDisconnect_v7000)(CUeglStreamConnection CUDAAPI *conn);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamProducerPresentFrame_v7000)(CUeglStreamConnection CUDAAPI *conn, CUeglFrame_v1 eglframe, CUstream CUDAAPI *pStream);
typedef CUresult (CUDAAPI *PFN_cuEGLStreamProducerReturnFrame_v7000)(CUeglStreamConnection CUDAAPI *conn, CUeglFrame_v1 CUDAAPI *eglframe, CUstream CUDAAPI *pStream);
typedef CUresult (CUDAAPI *PFN_cuGraphicsResourceGetMappedEglFrame_v7000)(CUeglFrame_v1 CUDAAPI *eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel);
typedef CUresult (CUDAAPI *PFN_cuEventCreateFromEGLSync_v9000)(CUevent CUDAAPI *phEvent, EGLSyncKHR eglSync, unsigned int flags);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
