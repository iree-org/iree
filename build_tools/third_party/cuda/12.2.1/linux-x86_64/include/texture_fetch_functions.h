/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__TEXTURE_FETCH_FUNCTIONS_H__)
#define __TEXTURE_FETCH_FUNCTIONS_H__


#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"
#include "cuda_texture_types.h"

#if defined(_WIN32)
# define __DEPRECATED__ __declspec(deprecated)
#else
# define __DEPRECATED__  __attribute__((deprecated))
#endif


template <typename T>
struct __nv_tex_rmet_ret { };

template<> struct __nv_tex_rmet_ret<char> { typedef char type; };
template<> struct __nv_tex_rmet_ret<signed char> { typedef signed char type; };
template<> struct __nv_tex_rmet_ret<unsigned char> { typedef unsigned char type; };
template<> struct __nv_tex_rmet_ret<char1> { typedef char1 type; };
template<> struct __nv_tex_rmet_ret<uchar1> { typedef uchar1 type; };
template<> struct __nv_tex_rmet_ret<char2> { typedef char2 type; };
template<> struct __nv_tex_rmet_ret<uchar2> { typedef uchar2 type; };
template<> struct __nv_tex_rmet_ret<char4> { typedef char4 type; };
template<> struct __nv_tex_rmet_ret<uchar4> { typedef uchar4 type; };

template<> struct __nv_tex_rmet_ret<short> { typedef short type; };
template<> struct __nv_tex_rmet_ret<unsigned short> { typedef unsigned short type; };
template<> struct __nv_tex_rmet_ret<short1> { typedef short1 type; };
template<> struct __nv_tex_rmet_ret<ushort1> { typedef ushort1 type; };
template<> struct __nv_tex_rmet_ret<short2> { typedef short2 type; };
template<> struct __nv_tex_rmet_ret<ushort2> { typedef ushort2 type; };
template<> struct __nv_tex_rmet_ret<short4> { typedef short4 type; };
template<> struct __nv_tex_rmet_ret<ushort4> { typedef ushort4 type; };

template<> struct __nv_tex_rmet_ret<int> { typedef int type; };
template<> struct __nv_tex_rmet_ret<unsigned int> { typedef unsigned int type; };
template<> struct __nv_tex_rmet_ret<int1> { typedef int1 type; };
template<> struct __nv_tex_rmet_ret<uint1> { typedef uint1 type; };
template<> struct __nv_tex_rmet_ret<int2> { typedef int2 type; };
template<> struct __nv_tex_rmet_ret<uint2> { typedef uint2 type; };
template<> struct __nv_tex_rmet_ret<int4> { typedef int4 type; };
template<> struct __nv_tex_rmet_ret<uint4> { typedef uint4 type; };

#if !defined(__LP64__)
template<> struct __nv_tex_rmet_ret<long> { typedef long type; };
template<> struct __nv_tex_rmet_ret<unsigned long> { typedef unsigned long type; };
template<> struct __nv_tex_rmet_ret<long1> { typedef long1 type; };
template<> struct __nv_tex_rmet_ret<ulong1> { typedef ulong1 type; };
template<> struct __nv_tex_rmet_ret<long2> { typedef long2 type; };
template<> struct __nv_tex_rmet_ret<ulong2> { typedef ulong2 type; };
template<> struct __nv_tex_rmet_ret<long4> { typedef long4 type; };
template<> struct __nv_tex_rmet_ret<ulong4> { typedef ulong4 type; };
#endif /* !__LP64__ */
template<> struct __nv_tex_rmet_ret<float> { typedef float type; };
template<> struct __nv_tex_rmet_ret<float1> { typedef float1 type; };
template<> struct __nv_tex_rmet_ret<float2> { typedef float2 type; };
template<> struct __nv_tex_rmet_ret<float4> { typedef float4 type; };


template <typename T> struct __nv_tex_rmet_cast { typedef T* type;  };
#if !defined(__LP64__)
template<> struct __nv_tex_rmet_cast<long> { typedef int *type; };
template<> struct __nv_tex_rmet_cast<unsigned long> { typedef unsigned int *type; };
template<> struct __nv_tex_rmet_cast<long1> { typedef int1 *type; };
template<> struct __nv_tex_rmet_cast<ulong1> { typedef uint1 *type; };
template<> struct __nv_tex_rmet_cast<long2> { typedef int2 *type; };
template<> struct __nv_tex_rmet_cast<ulong2> { typedef uint2 *type; };
template<> struct __nv_tex_rmet_cast<long4> { typedef int4 *type; };
template<> struct __nv_tex_rmet_cast<ulong4> { typedef uint4 *type; };
#endif /* !__LP64__ */

template <typename T>
struct __nv_tex_rmnf_ret { };

template <> struct __nv_tex_rmnf_ret<char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<signed char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<char1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<uchar1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<short1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<ushort1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<char2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<uchar2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<short2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<ushort2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<char4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<uchar4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<short4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<ushort4> { typedef float4 type; };


template <typename T>
struct __nv_tex2dgather_ret { };
template <> struct __nv_tex2dgather_ret<char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<signed char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char1> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char2> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char3> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char4> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<unsigned char> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar1> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar2> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar3> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar4> { typedef uchar4 type; };

template <> struct __nv_tex2dgather_ret<short> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short1> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short2> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short3> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short4> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<unsigned short> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort1> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort2> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort3> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort4> { typedef ushort4 type; };

template <> struct __nv_tex2dgather_ret<int> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int1> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int2> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int3> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int4> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<unsigned int> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint1> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint2> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint3> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint4> { typedef uint4 type; };

template <> struct __nv_tex2dgather_ret<float> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float1> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float2> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float3> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float4> { typedef float4 type; };


template<typename T> struct __nv_tex2dgather_rmnf_ret { };
template<> struct __nv_tex2dgather_rmnf_ret<char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<signed char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<unsigned char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<signed short> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<unsigned short> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort4> { typedef float4 type; };

#undef __DEPRECATED__

#endif /* __cplusplus && __CUDACC__ */

#endif /* !__TEXTURE_FETCH_FUNCTIONS_H__ */
