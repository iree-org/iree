/*
 * Copyright 2014 NVIDIA Corporation.  All rights reserved.
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

#ifndef CUDAEGL_H
#define CUDAEGL_H

#include "cuda.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"


#ifdef CUDA_FORCE_API_VERSION
#error "CUDA_FORCE_API_VERSION is no longer supported."
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
  * \addtogroup CUDA_TYPES
  * @{
  */

/**
 * Maximum number of planes per frame
 */
#define MAX_PLANES 3

/**
  * CUDA EglFrame type - array or pointer
  */
typedef enum CUeglFrameType_enum {
    CU_EGL_FRAME_TYPE_ARRAY = 0,  /**< Frame type CUDA array */
    CU_EGL_FRAME_TYPE_PITCH = 1,  /**< Frame type pointer */
} CUeglFrameType;

/**
 * Indicates that timeout for ::cuEGLStreamConsumerAcquireFrame is infinite.
 */
#define CUDA_EGL_INFINITE_TIMEOUT 0xFFFFFFFF

/**
 * Resource location flags- sysmem or vidmem
 *
 * For CUDA context on iGPU, since video and system memory are equivalent -
 * these flags will not have an effect on the execution.
 *
 * For CUDA context on dGPU, applications can use the flag ::CUeglResourceLocationFlags
 * to give a hint about the desired location.
 *
 * ::CU_EGL_RESOURCE_LOCATION_SYSMEM - the frame data is made resident on the system memory
 * to be accessed by CUDA.
 *
 * ::CU_EGL_RESOURCE_LOCATION_VIDMEM - the frame data is made resident on the dedicated
 * video memory to be accessed by CUDA.
 *
 * There may be an additional latency due to new allocation and data migration,
 * if the frame is produced on a different memory.

  */
typedef enum CUeglResourceLocationFlags_enum {
    CU_EGL_RESOURCE_LOCATION_SYSMEM   = 0x00,       /**< Resource location sysmem */
    CU_EGL_RESOURCE_LOCATION_VIDMEM   = 0x01        /**< Resource location vidmem */
} CUeglResourceLocationFlags;

/**
  * CUDA EGL Color Format - The different planar and multiplanar formats currently supported for CUDA_EGL interops.
  * Three channel formats are currently not supported for ::CU_EGL_FRAME_TYPE_ARRAY
  */
typedef enum CUeglColorFormat_enum {
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR              = 0x00,  /**< Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR          = 0x01,  /**< Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV420Planar. */
    CU_EGL_COLOR_FORMAT_YUV422_PLANAR              = 0x02,  /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR          = 0x03,  /**< Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar. */
    CU_EGL_COLOR_FORMAT_RGB                        = 0x04,  /**< R/G/B three channels in one surface with BGR byte ordering. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_BGR                        = 0x05,  /**< R/G/B three channels in one surface with RGB byte ordering. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_ARGB                       = 0x06,  /**< R/G/B/A four channels in one surface with BGRA byte ordering. */
    CU_EGL_COLOR_FORMAT_RGBA                       = 0x07,  /**< R/G/B/A four channels in one surface with ABGR byte ordering. */
    CU_EGL_COLOR_FORMAT_L                          = 0x08,  /**< single luminance channel in one surface. */
    CU_EGL_COLOR_FORMAT_R                          = 0x09,  /**< single color channel in one surface. */
    CU_EGL_COLOR_FORMAT_YUV444_PLANAR              = 0x0A,  /**< Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR          = 0x0B,  /**< Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV444Planar. */
    CU_EGL_COLOR_FORMAT_YUYV_422                   = 0x0C,  /**< Y, U, V in one surface, interleaved as UYVY in one channel. */
    CU_EGL_COLOR_FORMAT_UYVY_422                   = 0x0D,  /**< Y, U, V in one surface, interleaved as YUYV in one channel. */
    CU_EGL_COLOR_FORMAT_ABGR                       = 0x0E,  /**< R/G/B/A four channels in one surface with RGBA byte ordering. */
    CU_EGL_COLOR_FORMAT_BGRA                       = 0x0F,  /**< R/G/B/A four channels in one surface with ARGB byte ordering. */
    CU_EGL_COLOR_FORMAT_A                          = 0x10,  /**< Alpha color format - one channel in one surface. */
    CU_EGL_COLOR_FORMAT_RG                         = 0x11,  /**< R/G color format - two channels in one surface with GR byte ordering */
    CU_EGL_COLOR_FORMAT_AYUV                       = 0x12,  /**< Y, U, V, A four channels in one surface, interleaved as VUYA. */
    CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR          = 0x13,  /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR          = 0x14,  /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR          = 0x15,  /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR   = 0x16,  /**< Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR   = 0x17,  /**< Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR   = 0x18,  /**< Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR   = 0x19,  /**< Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_VYUY_ER                    = 0x1A,  /**< Extended Range Y, U, V in one surface, interleaved as YVYU in one channel. */
    CU_EGL_COLOR_FORMAT_UYVY_ER                    = 0x1B,  /**< Extended Range Y, U, V in one surface, interleaved as YUYV in one channel. */
    CU_EGL_COLOR_FORMAT_YUYV_ER                    = 0x1C,  /**< Extended Range Y, U, V in one surface, interleaved as UYVY in one channel. */
    CU_EGL_COLOR_FORMAT_YVYU_ER                    = 0x1D,  /**< Extended Range Y, U, V in one surface, interleaved as VYUY in one channel. */
    CU_EGL_COLOR_FORMAT_YUV_ER                     = 0x1E,  /**< Extended Range Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_YUVA_ER                    = 0x1F,  /**< Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY. */
    CU_EGL_COLOR_FORMAT_AYUV_ER                    = 0x20,  /**< Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA. */
    CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER           = 0x21,  /**< Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER           = 0x22,  /**< Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER           = 0x23,  /**< Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER       = 0x24,  /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER       = 0x25,  /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER       = 0x26,  /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER           = 0x27,  /**< Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER           = 0x28,  /**< Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER           = 0x29,  /**< Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER       = 0x2A,  /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER       = 0x2B,  /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER       = 0x2C,  /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_BAYER_RGGB                 = 0x2D,  /**< Bayer format - one channel in one surface with interleaved RGGB ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_BGGR                 = 0x2E,  /**< Bayer format - one channel in one surface with interleaved BGGR ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_GRBG                 = 0x2F,  /**< Bayer format - one channel in one surface with interleaved GRBG ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_GBRG                 = 0x30,  /**< Bayer format - one channel in one surface with interleaved GBRG ordering. */
    CU_EGL_COLOR_FORMAT_BAYER10_RGGB               = 0x31,  /**< Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_BGGR               = 0x32,  /**< Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_GRBG               = 0x33,  /**< Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_GBRG               = 0x34,  /**< Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_RGGB               = 0x35,  /**< Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_BGGR               = 0x36,  /**< Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_GRBG               = 0x37,  /**< Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_GBRG               = 0x38,  /**< Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_RGGB               = 0x39,  /**< Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_BGGR               = 0x3A,  /**< Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_GRBG               = 0x3B,  /**< Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_GBRG               = 0x3C,  /**< Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_RGGB               = 0x3D,  /**< Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_BGGR               = 0x3E,  /**< Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_GRBG               = 0x3F,  /**< Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_GBRG               = 0x40,  /**< Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_YVU444_PLANAR              = 0x41,  /**< Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_PLANAR              = 0x42,  /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR              = 0x43,  /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB             = 0x44,  /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR             = 0x45,  /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG             = 0x46,  /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG             = 0x47,  /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_BCCR                 = 0x48,  /**< Bayer format - one channel in one surface with interleaved BCCR ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_RCCB                 = 0x49,  /**< Bayer format - one channel in one surface with interleaved RCCB ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_CRBC                 = 0x4A,  /**< Bayer format - one channel in one surface with interleaved CRBC ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_CBRC                 = 0x4B,  /**< Bayer format - one channel in one surface with interleaved CBRC ordering. */
    CU_EGL_COLOR_FORMAT_BAYER10_CCCC               = 0x4C,  /**< Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_BCCR               = 0x4D,  /**< Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_RCCB               = 0x4E,  /**< Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_CRBC               = 0x4F,  /**< Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_CBRC               = 0x50,  /**< Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_CCCC               = 0x51,  /**< Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_Y                          = 0x52, /**< Color format for single Y plane. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020     = 0x53, /**< Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020     = 0x54, /**< Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020         = 0x55, /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height= 1/2 Y height. */             
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020         = 0x56, /**< Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height
= 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709      = 0x57, /**< Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709      = 0x58, /**< Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709          = 0x59, /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height
= 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709          = 0x5A,  /**< Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709  = 0x5B, /**< Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 = 0x5C, /**< Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 = 0x5D, /**< Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height  = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR      = 0x5E, /**< Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height  = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709  = 0x5F, /**< Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height  = Y height. */
    CU_EGL_COLOR_FORMAT_Y_ER                          = 0x60, /**< Extended Range Color format for single Y plane. */
    CU_EGL_COLOR_FORMAT_Y_709_ER                      = 0x61, /**< Extended Range Color format for single Y plane. */
    CU_EGL_COLOR_FORMAT_Y10_ER                        = 0x62, /**< Extended Range Color format for single Y10 plane. */
    CU_EGL_COLOR_FORMAT_Y10_709_ER                    = 0x63, /**< Extended Range Color format for single Y10 plane. */
    CU_EGL_COLOR_FORMAT_Y12_ER                        = 0x64, /**< Extended Range Color format for single Y12 plane. */
    CU_EGL_COLOR_FORMAT_Y12_709_ER                    = 0x65, /**< Extended Range Color format for single Y12 plane. */
    CU_EGL_COLOR_FORMAT_YUVA                          = 0x66, /**< Y, U, V, A four channels in one surface, interleaved as AVUY. */
    CU_EGL_COLOR_FORMAT_YUV                           = 0x67, /**< Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_YVYU                          = 0x68, /**< Y, U, V in one surface, interleaved as YVYU in one channel. */
    CU_EGL_COLOR_FORMAT_VYUY                          = 0x69, /**< Y, U, V in one surface, interleaved as VYUY in one channel. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER     = 0x6A, /**< Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER = 0x6B, /**< Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER     = 0x6C, /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */ 
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER = 0x6D, /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface)  U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER     = 0x6E, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */ 
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER = 0x6F, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER     = 0x70, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */ 
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER = 0x71, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_MAX
} CUeglColorFormat;

/**
 * CUDA EGLFrame structure Descriptor - structure defining one frame of EGL.
 *
 * Each frame may contain one or more planes depending on whether the surface  * is Multiplanar or not.
 */
typedef struct CUeglFrame_st {
    union {
        CUarray pArray[MAX_PLANES];     /**< Array of CUarray corresponding to each plane*/
        void*   pPitch[MAX_PLANES];     /**< Array of Pointers corresponding to each plane*/
    } frame;
    unsigned int width;                 /**< Width of first plane */
    unsigned int height;                /**< Height of first plane */
    unsigned int depth;                 /**< Depth of first plane */
    unsigned int pitch;                 /**< Pitch of first plane */
    unsigned int planeCount;            /**< Number of planes */
    unsigned int numChannels;           /**< Number of channels for the plane */
    CUeglFrameType frameType;           /**< Array or Pitch */
    CUeglColorFormat eglColorFormat;    /**< CUDA EGL Color Format*/
    CUarray_format cuFormat;            /**< CUDA Array Format*/
} CUeglFrame_v1;
typedef CUeglFrame_v1 CUeglFrame;

/**
  * CUDA EGLSream Connection
  */
typedef struct CUeglStreamConnection_st* CUeglStreamConnection;

/** @} */ /* END CUDA_TYPES */

/**
 * \file cudaEGL.h
 * \brief Header file for the EGL interoperability functions of the
 * low-level CUDA driver application programming interface.
 */

/**
 * \defgroup CUDA_EGL EGL Interoperability
 * \ingroup CUDA_DRIVER
 *
 * ___MANBRIEF___ EGL interoperability functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the EGL interoperability functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Registers an EGL image
 *
 * Registers the EGLImageKHR specified by \p image for access by
 * CUDA. A handle to the registered object is returned as \p pCudaResource.
 * Additional Mapping/Unmapping is not required for the registered resource and
 * ::cuGraphicsResourceGetMappedEglFrame can be directly called on the \p pCudaResource.
 *
 * The application will be responsible for synchronizing access to shared objects.
 * The application must ensure that any pending operation which access the objects have completed
 * before passing control to CUDA. This may be accomplished by issuing and waiting for
 * glFinish command on all GLcontexts (for OpenGL and likewise for other APIs).
 * The application will be also responsible for ensuring that any pending operation on the
 * registered CUDA resource has completed prior to executing subsequent commands in other APIs
 * accesing the same memory objects.
 * This can be accomplished by calling cuCtxSynchronize or cuEventSynchronize (preferably).
 *
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
 * The EGLImageKHR is an object which can be used to create EGLImage target resource. It is defined as a void pointer.
 * typedef void* EGLImageKHR
 *
 * \param pCudaResource   - Pointer to the returned object handle
 * \param image           - An EGLImageKHR image which can be used to create target resource.
 * \param flags           - Map flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 *
 * \sa ::cuGraphicsEGLRegisterImage, ::cuGraphicsUnregisterResource,
 * ::cuGraphicsResourceSetMapFlags, ::cuGraphicsMapResources,
 * ::cuGraphicsUnmapResources,
 * ::cudaGraphicsEGLRegisterImage
 */
CUresult CUDAAPI cuGraphicsEGLRegisterImage(CUgraphicsResource *pCudaResource, EGLImageKHR image, unsigned int flags);

/**
 * \brief Connect CUDA to EGLStream as a consumer.
 *
 * Connect CUDA as a consumer to EGLStreamKHR specified by \p stream.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * \param conn            - Pointer to the returned connection handle
 * \param stream          - EGLStreamKHR handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 *
 * \sa ::cuEGLStreamConsumerConnect, ::cuEGLStreamConsumerDisconnect,
 * ::cuEGLStreamConsumerAcquireFrame, ::cuEGLStreamConsumerReleaseFrame,
 * ::cudaEGLStreamConsumerConnect
 */
CUresult CUDAAPI cuEGLStreamConsumerConnect(CUeglStreamConnection *conn, EGLStreamKHR stream);

/**
 * \brief Connect CUDA to EGLStream as a consumer with given flags.
 *
 * Connect CUDA as a consumer to EGLStreamKHR specified by \p stream with specified \p flags defined by CUeglResourceLocationFlags.
 *
 * The flags specify whether the consumer wants to access frames from system memory or video memory.
 * Default is ::CU_EGL_RESOURCE_LOCATION_VIDMEM.
 *
 * \param conn              - Pointer to the returned connection handle
 * \param stream            - EGLStreamKHR handle
 * \param flags             - Flags denote intended location - system or video.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 *
 * \sa ::cuEGLStreamConsumerConnect, ::cuEGLStreamConsumerDisconnect,
 * ::cuEGLStreamConsumerAcquireFrame, ::cuEGLStreamConsumerReleaseFrame,
 * ::cudaEGLStreamConsumerConnectWithFlags
 */

CUresult CUDAAPI cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection *conn, EGLStreamKHR stream, unsigned int flags);

/**
 * \brief Disconnect CUDA as a consumer to EGLStream .
 *
 * Disconnect CUDA as a consumer to EGLStreamKHR.
 *
 * \param conn            - Conection to disconnect.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 *
 * \sa ::cuEGLStreamConsumerConnect, ::cuEGLStreamConsumerDisconnect,
 * ::cuEGLStreamConsumerAcquireFrame, ::cuEGLStreamConsumerReleaseFrame,
 * ::cudaEGLStreamConsumerDisconnect
 */
CUresult CUDAAPI cuEGLStreamConsumerDisconnect(CUeglStreamConnection *conn);

/**
 * \brief Acquire an image frame from the EGLStream with CUDA as a consumer.
 *
 * Acquire an image frame from EGLStreamKHR. This API can also acquire an old frame presented
 * by the producer unless explicitly disabled by setting EGL_SUPPORT_REUSE_NV flag to EGL_FALSE
 * during stream initialization. By default, EGLStream is created with this flag set to EGL_TRUE.
 * ::cuGraphicsResourceGetMappedEglFrame can be called on \p pCudaResource to get
 * ::CUeglFrame.
 *
 * \param conn            - Connection on which to acquire
 * \param pCudaResource   - CUDA resource on which the stream frame will be mapped for use.
 * \param pStream         - CUDA stream for synchronization and any data migrations
 *                          implied by ::CUeglResourceLocationFlags.
 * \param timeout         - Desired timeout in usec for a new frame to be acquired.
 *                          If set as ::CUDA_EGL_INFINITE_TIMEOUT, acquire waits infinitely.
 *                          After timeout occurs CUDA consumer tries to acquire an old frame
 *                          if available and EGL_SUPPORT_REUSE_NV flag is set.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 *
 * \sa ::cuEGLStreamConsumerConnect, ::cuEGLStreamConsumerDisconnect,
 * ::cuEGLStreamConsumerAcquireFrame, ::cuEGLStreamConsumerReleaseFrame,
 * ::cudaEGLStreamConsumerAcquireFrame
 */
CUresult CUDAAPI cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection *conn,
                                                  CUgraphicsResource *pCudaResource, CUstream *pStream, unsigned int timeout);
/**
 * \brief Releases the last frame acquired from the EGLStream.
 *
 * Release the acquired image frame specified by \p pCudaResource to EGLStreamKHR.
 * If EGL_SUPPORT_REUSE_NV flag is set to EGL_TRUE, at the time of EGL creation
 * this API doesn't release the last frame acquired on the EGLStream.
 * By default, EGLStream is created with this flag set to EGL_TRUE.
 *
 * \param conn            - Connection on which to release
 * \param pCudaResource   - CUDA resource whose corresponding frame is to be released
 * \param pStream         - CUDA stream on which release will be done.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 *
 * \sa ::cuEGLStreamConsumerConnect, ::cuEGLStreamConsumerDisconnect,
 * ::cuEGLStreamConsumerAcquireFrame, ::cuEGLStreamConsumerReleaseFrame,
 * ::cudaEGLStreamConsumerReleaseFrame
 */
CUresult CUDAAPI cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection *conn,
                                                  CUgraphicsResource pCudaResource, CUstream *pStream);

/**
 * \brief Connect CUDA to EGLStream as a producer.
 *
 * Connect CUDA as a producer to EGLStreamKHR specified by \p stream.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * \param conn   - Pointer to the returned connection handle
 * \param stream - EGLStreamKHR handle
 * \param width  - width of the image to be submitted to the stream
 * \param height - height of the image to be submitted to the stream
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 *
 * \sa ::cuEGLStreamProducerConnect, ::cuEGLStreamProducerDisconnect,
 * ::cuEGLStreamProducerPresentFrame,
 * ::cudaEGLStreamProducerConnect
 */
CUresult CUDAAPI cuEGLStreamProducerConnect(CUeglStreamConnection *conn, EGLStreamKHR stream,
                                             EGLint width, EGLint height);

/**
 * \brief Disconnect CUDA as a producer  to EGLStream .
 *
 * Disconnect CUDA as a producer to EGLStreamKHR.
 *
 * \param conn            - Conection to disconnect.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 *
 * \sa ::cuEGLStreamProducerConnect, ::cuEGLStreamProducerDisconnect,
 * ::cuEGLStreamProducerPresentFrame,
 * ::cudaEGLStreamProducerDisconnect
 */
CUresult CUDAAPI cuEGLStreamProducerDisconnect(CUeglStreamConnection *conn);

/**
 * \brief Present a CUDA eglFrame to the EGLStream with CUDA as a producer.
 *
 * When a frame is presented by the producer, it gets associated with the EGLStream
 * and thus it is illegal to free the frame before the producer is disconnected.
 * If a frame is freed and reused it may lead to undefined behavior.
 *
 * If producer and consumer are on different GPUs (iGPU and dGPU) then frametype
 * ::CU_EGL_FRAME_TYPE_ARRAY is not supported. ::CU_EGL_FRAME_TYPE_PITCH can be used for
 * such cross-device applications.
 *
 * The ::CUeglFrame is defined as:
 * \code
 * typedef struct CUeglFrame_st {
 *     union {
 *         CUarray pArray[MAX_PLANES];
 *         void*   pPitch[MAX_PLANES];
 *     } frame;
 *     unsigned int width;
 *     unsigned int height;
 *     unsigned int depth;
 *     unsigned int pitch;
 *     unsigned int planeCount;
 *     unsigned int numChannels;
 *     CUeglFrameType frameType;
 *     CUeglColorFormat eglColorFormat;
 *     CUarray_format cuFormat;
 * } CUeglFrame;
 * \endcode
 *
 * For ::CUeglFrame of type ::CU_EGL_FRAME_TYPE_PITCH, the application may present sub-region of a memory
 * allocation. In that case, the pitched pointer will specify the start address of the sub-region in
 * the allocation and corresponding ::CUeglFrame fields will specify the dimensions of the sub-region.
 * 
 * \param conn            - Connection on which to present the CUDA array
 * \param eglframe        - CUDA Eglstream Proucer Frame handle to be sent to the consumer over EglStream.
 * \param pStream         - CUDA stream on which to present the frame.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 *
 * \sa ::cuEGLStreamProducerConnect, ::cuEGLStreamProducerDisconnect,
 * ::cuEGLStreamProducerReturnFrame,
 * ::cudaEGLStreamProducerPresentFrame
 */
CUresult CUDAAPI cuEGLStreamProducerPresentFrame(CUeglStreamConnection *conn,
                                                 CUeglFrame eglframe, CUstream *pStream);

/**
 * \brief Return the CUDA eglFrame to the EGLStream released by the consumer.
 *
 * This API can potentially return CUDA_ERROR_LAUNCH_TIMEOUT if the consumer has not 
 * returned a frame to EGL stream. If timeout is returned the application can retry.
 *
 * \param conn            - Connection on which to return
 * \param eglframe        - CUDA Eglstream Proucer Frame handle returned from the consumer over EglStream.
 * \param pStream         - CUDA stream on which to return the frame.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT
 *
 * \sa ::cuEGLStreamProducerConnect, ::cuEGLStreamProducerDisconnect,
 * ::cuEGLStreamProducerPresentFrame,
 * ::cudaEGLStreamProducerReturnFrame
 */
CUresult CUDAAPI cuEGLStreamProducerReturnFrame(CUeglStreamConnection *conn,
                                                CUeglFrame *eglframe, CUstream *pStream);

/**
 * \brief Get an eglFrame through which to access a registered EGL graphics resource.
 *
 * Returns in \p *eglFrame an eglFrame pointer through which the registered graphics resource
 * \p resource may be accessed.
 * This API can only be called for registered EGL graphics resources.
 *
 * The ::CUeglFrame is defined as:
 * \code
 * typedef struct CUeglFrame_st {
 *     union {
 *         CUarray pArray[MAX_PLANES];
 *         void*   pPitch[MAX_PLANES];
 *     } frame;
 *     unsigned int width;
 *     unsigned int height;
 *     unsigned int depth;
 *     unsigned int pitch;
 *     unsigned int planeCount;
 *     unsigned int numChannels;
 *     CUeglFrameType frameType;
 *     CUeglColorFormat eglColorFormat;
 *     CUarray_format cuFormat;
 * } CUeglFrame;
 * \endcode
 *
 * If \p resource is not registered then ::CUDA_ERROR_NOT_MAPPED is returned.
 * *
 * \param eglFrame   - Returned eglFrame.
 * \param resource   - Registered resource to access.
 * \param index      - Index for cubemap surfaces.
 * \param mipLevel   - Mipmap level for the subresource to access.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED
 *
 * \sa
 * ::cuGraphicsMapResources,
 * ::cuGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsResourceGetMappedPointer,
 * ::cudaGraphicsResourceGetMappedEglFrame
 */
CUresult CUDAAPI cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel);

/**
 * \brief Creates an event from EGLSync object
 *
 * Creates an event *phEvent from an EGLSyncKHR eglSync with the flags specified
 * via \p flags. Valid flags include:
 * - ::CU_EVENT_DEFAULT: Default event creation flag.
 * - ::CU_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking
 * synchronization.  A CPU thread that uses ::cuEventSynchronize() to wait on
 * an event created with this flag will block until the event has actually
 * been completed.
 *
 * Once the \p eglSync gets destroyed, ::cuEventDestroy is the only API
 * that can be invoked on the event.
 *
 * ::cuEventRecord and TimingData are not supported for events created from EGLSync.
 *
 * The EGLSyncKHR is an opaque handle to an EGL sync object.
 * typedef void* EGLSyncKHR
 *
 * \param phEvent - Returns newly created event
 * \param eglSync - Opaque handle to EGLSync object
 * \param flags   - Event creation flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa
 * ::cuEventQuery,
 * ::cuEventSynchronize,
 * ::cuEventDestroy
 */
CUresult CUDAAPI cuEventCreateFromEGLSync(CUevent *phEvent, EGLSyncKHR eglSync, unsigned int flags);

/** @} */ /* END CUDA_EGL */

#ifdef __cplusplus
};
#endif

#endif

