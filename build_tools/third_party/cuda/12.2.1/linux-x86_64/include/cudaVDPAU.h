/*
 * Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
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

#ifndef CUDAVDPAU_H
#define CUDAVDPAU_H

#ifdef CUDA_FORCE_API_VERSION
#error "CUDA_FORCE_API_VERSION is no longer supported."
#endif

#define cuVDPAUCtxCreate cuVDPAUCtxCreate_v2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup CUDA_VDPAU VDPAU Interoperability
 * \ingroup CUDA_DRIVER
 *
 * ___MANBRIEF___ VDPAU interoperability functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the VDPAU interoperability functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Gets the CUDA device associated with a VDPAU device
 *
 * Returns in \p *pDevice the CUDA device associated with a \p vdpDevice, if
 * applicable.
 *
 * \param pDevice           - Device associated with vdpDevice
 * \param vdpDevice         - A VdpDevice handle
 * \param vdpGetProcAddress - VDPAU's VdpGetProcAddress function pointer
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate, ::cuVDPAUCtxCreate, ::cuGraphicsVDPAURegisterVideoSurface,
 * ::cuGraphicsVDPAURegisterOutputSurface, ::cuGraphicsUnregisterResource,
 * ::cuGraphicsResourceSetMapFlags, ::cuGraphicsMapResources,
 * ::cuGraphicsUnmapResources, ::cuGraphicsSubResourceGetMappedArray,
 * ::cudaVDPAUGetDevice
 */
CUresult CUDAAPI cuVDPAUGetDevice(CUdevice *pDevice, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);

/**
 * \brief Create a CUDA context for interoperability with VDPAU
 *
 * Creates a new CUDA context, initializes VDPAU interoperability, and
 * associates the CUDA context with the calling thread. It must be called
 * before performing any other VDPAU interoperability operations. It may fail
 * if the needed VDPAU driver facilities are not available. For usage of the
 * \p flags parameter, see ::cuCtxCreate().
 *
 * \param pCtx              - Returned CUDA context
 * \param flags             - Options for CUDA context creation
 * \param device            - Device on which to create the context
 * \param vdpDevice         - The VdpDevice to interop with
 * \param vdpGetProcAddress - VDPAU's VdpGetProcAddress function pointer
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuCtxCreate, ::cuGraphicsVDPAURegisterVideoSurface,
 * ::cuGraphicsVDPAURegisterOutputSurface, ::cuGraphicsUnregisterResource,
 * ::cuGraphicsResourceSetMapFlags, ::cuGraphicsMapResources,
 * ::cuGraphicsUnmapResources, ::cuGraphicsSubResourceGetMappedArray,
 * ::cuVDPAUGetDevice
 */
CUresult CUDAAPI cuVDPAUCtxCreate(CUcontext *pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);

/**
 * \brief Registers a VDPAU VdpVideoSurface object
 *
 * Registers the VdpVideoSurface specified by \p vdpSurface for access by
 * CUDA. A handle to the registered object is returned as \p pCudaResource.
 * The surface's intended usage is specified using \p flags, as follows:
 *
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA
 *   will not write to this resource.
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * The VdpVideoSurface is presented as an array of subresources that may be
 * accessed using pointers returned by ::cuGraphicsSubResourceGetMappedArray.
 * The exact number of valid \p arrayIndex values depends on the VDPAU surface
 * format. The mapping is shown in the table below. \p mipLevel must be 0.
 *
 * \htmlonly
 * <table>
 * <tr><th>VdpChromaType                               </th><th>arrayIndex</th><th>Size     </th><th>Format</th><th>Content            </th></tr>
 * <tr><td rowspan="4" valign="top">VDP_CHROMA_TYPE_420</td><td>0         </td><td>w   x h/2</td><td>R8    </td><td>Top-field luma     </td></tr>
 * <tr>                                                     <td>1         </td><td>w   x h/2</td><td>R8    </td><td>Bottom-field luma  </td></tr>
 * <tr>                                                     <td>2         </td><td>w/2 x h/4</td><td>R8G8  </td><td>Top-field chroma   </td></tr>
 * <tr>                                                     <td>3         </td><td>w/2 x h/4</td><td>R8G8  </td><td>Bottom-field chroma</td></tr>
 * <tr><td rowspan="4" valign="top">VDP_CHROMA_TYPE_422</td><td>0         </td><td>w   x h/2</td><td>R8    </td><td>Top-field luma     </td></tr>
 * <tr>                                                     <td>1         </td><td>w   x h/2</td><td>R8    </td><td>Bottom-field luma  </td></tr>
 * <tr>                                                     <td>2         </td><td>w/2 x h/2</td><td>R8G8  </td><td>Top-field chroma   </td></tr>
 * <tr>                                                     <td>3         </td><td>w/2 x h/2</td><td>R8G8  </td><td>Bottom-field chroma</td></tr>
 * </table>
 * \endhtmlonly
 *
 * \latexonly
 * \begin{tabular}{|l|l|l|l|l|}
 * \hline
 * VdpChromaType          & arrayIndex & Size      & Format & Content             \\
 * \hline
 * VDP\_CHROMA\_TYPE\_420 & 0          & w x h/2   & R8     & Top-field luma      \\
 *                        & 1          & w x h/2   & R8     & Bottom-field luma   \\
 *                        & 2          & w/2 x h/4 & R8G8   & Top-field chroma    \\
 *                        & 3          & w/2 x h/4 & R8G8   & Bottom-field chroma \\
 * \hline
 * VDP\_CHROMA\_TYPE\_422 & 0          & w x h/2   & R8     & Top-field luma      \\
 *                        & 1          & w x h/2   & R8     & Bottom-field luma   \\
 *                        & 2          & w/2 x h/2 & R8G8   & Top-field chroma    \\
 *                        & 3          & w/2 x h/2 & R8G8   & Bottom-field chroma \\
 * \hline
 * \end{tabular}
 * \endlatexonly
 *
 * \param pCudaResource - Pointer to the returned object handle
 * \param vdpSurface    - The VdpVideoSurface to be registered
 * \param flags         - Map flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * \notefnerr
 *
 * \sa ::cuCtxCreate, ::cuVDPAUCtxCreate,
 * ::cuGraphicsVDPAURegisterOutputSurface, ::cuGraphicsUnregisterResource,
 * ::cuGraphicsResourceSetMapFlags, ::cuGraphicsMapResources,
 * ::cuGraphicsUnmapResources, ::cuGraphicsSubResourceGetMappedArray,
 * ::cuVDPAUGetDevice,
 * ::cudaGraphicsVDPAURegisterVideoSurface
 */
CUresult CUDAAPI cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource *pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags);

/**
 * \brief Registers a VDPAU VdpOutputSurface object
 *
 * Registers the VdpOutputSurface specified by \p vdpSurface for access by
 * CUDA. A handle to the registered object is returned as \p pCudaResource.
 * The surface's intended usage is specified using \p flags, as follows:
 *
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA
 *   will not write to this resource.
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * The VdpOutputSurface is presented as an array of subresources that may be
 * accessed using pointers returned by ::cuGraphicsSubResourceGetMappedArray.
 * The exact number of valid \p arrayIndex values depends on the VDPAU surface
 * format. The mapping is shown in the table below. \p mipLevel must be 0.
 *
 * \htmlonly
 * <table>
 * <tr><th>VdpRGBAFormat              </th><th>arrayIndex</th><th>Size </th><th>Format </th><th>Content       </th></tr>
 * <tr><td>VDP_RGBA_FORMAT_B8G8R8A8   </td><td>0         </td><td>w x h</td><td>ARGB8  </td><td>Entire surface</td></tr>
 * <tr><td>VDP_RGBA_FORMAT_R10G10B10A2</td><td>0         </td><td>w x h</td><td>A2BGR10</td><td>Entire surface</td></tr>
 * </table>
 * \endhtmlonly
 *
 * \latexonly
 * \begin{tabular}{|l|l|l|l|l|}
 * \hline
 * VdpRGBAFormat                  & arrayIndex & Size  & Format  & Content        \\
 * \hline
 * VDP\_RGBA\_FORMAT\_B8G8R8A8    & 0          & w x h & ARGB8   & Entire surface \\
 * VDP\_RGBA\_FORMAT\_R10G10B10A2 & 0          & w x h & A2BGR10 & Entire surface \\
 * \hline
 * \end{tabular}
 * \endlatexonly
 *
 * \param pCudaResource - Pointer to the returned object handle
 * \param vdpSurface    - The VdpOutputSurface to be registered
 * \param flags         - Map flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * \notefnerr
 *
 * \sa ::cuCtxCreate, ::cuVDPAUCtxCreate,
 * ::cuGraphicsVDPAURegisterVideoSurface, ::cuGraphicsUnregisterResource,
 * ::cuGraphicsResourceSetMapFlags, ::cuGraphicsMapResources,
 * ::cuGraphicsUnmapResources, ::cuGraphicsSubResourceGetMappedArray,
 * ::cuVDPAUGetDevice,
 * ::cudaGraphicsVDPAURegisterOutputSurface
 */
CUresult CUDAAPI cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource *pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags);

/** @} */ /* END CUDA_VDPAU */


#if defined(__CUDA_API_VERSION_INTERNAL)
    #undef cuVDPAUCtxCreate

    CUresult CUDAAPI cuVDPAUCtxCreate(CUcontext *pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
#endif /* __CUDA_API_VERSION_INTERNAL */

#ifdef __cplusplus
};
#endif

#endif
