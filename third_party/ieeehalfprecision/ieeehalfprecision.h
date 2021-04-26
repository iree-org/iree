/******************************************************************************
 *
 * Filename:    ieeehalfprecision.h
 * Programmer:  James Tursa
 * Version:     1.0
 * Date:        March 3, 2009
 * Copyright:   (c) 2009 by James Tursa, All Rights Reserved
 *
 *  This code uses the BSD License:
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are
 *  met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the distribution
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef IEEE_HALF_PRECISION_CONVERSION_UTIL_HPP_
#define IEEE_HALF_PRECISION_CONVERSION_UTIL_HPP_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


  /*!\fn singles2halfp
    Converts from 32bit-floats to 16bit-floats.
    [routine from James Tursa, BSD License,
     see ieeehalfprecision.c for details].
    \param target destination to which to write 16bit-floats
    \param source source from which to read 32bit-floats
    \param numel number of conversions to perform, i.e.
                 source points to numel floats
                 and target points to 2*numel bytes.
   */
  void singles2halfp(uint16_t *target, const uint32_t *source, int numel);


  /*!\fn halfp2singles
    Converts from 16bit-floats to 32bit-floats.
    [routine from James Tursa, BSD License,
     see ieeehalfprecision.c for details].
    \param target destination to which to write 32bit-floats
    \param source source from which to read 16bit-floats
    \param numel number of conversions to perform, i.e.
                 source points to 2*numel bytes
                 and target points to numem floats.
   */
  void halfp2singles(uint32_t *target, const uint16_t *source, int numel);

#ifdef __cplusplus
}
#endif

#endif
