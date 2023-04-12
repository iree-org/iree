/*
 * Copyright (c) 2012      Oracle and/or its affiliates.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OMPI_FORTRAN_STATUS_CONVERSION_H
#define OMPI_FORTRAN_STATUS_CONVERSION_H

#include "ompi_config.h"
#include "mpi.h"

/*
 * A Fortran MPI_STATUS argument and a C MPI_Status argument can differ.
 * Therefore, the Fortran layer converts between Fortran and C statuses
 * using the MPI_Status_[f2c|c2f] functions:
 *
 *   void Fortran_api(... MPI_Fint *status ...)
 *   {
 *       int c_ierr;
 *       MPI_Status   c_status;
 *       MPI_Status_f2c(status, &c_status);
 *       c_ierr = C_api(... &c_status ...);
 *       MPI_Status_c2f(&c_status, status);
 *   }
 *
 * The macros we define below support a different approach that avoids
 * the overhead of conversion in cases where we can detect that the
 * Fortran status can be used directly:
 *
 *   void Fortran_api(... MPI_Fint *status ...)
 *   {
 *       int c_ierr;
 *       OMPI_FORTRAN_STATUS_DECLARATION(c_status,c_status2)
 *       OMPI_FORTRAN_STATUS_SET_POINTER(c_status,c_status2,status)
 *       c_ierr = C_api(... c_status ...);
 *       OMPI_FORTRAN_STATUS_RETURN(c_status,c_status2,status,c_ierr)
 *   }
 *
 * Issues around whether a Fortran status can be used directly by
 * OMPI C internals are discussed in trac tickets 2526 and 3218 as
 * well as ompi/mpi/c/status_c2f.c.  The issues include:
 *
 * - A Fortran status must be large enough to hold a C status.
 *   This requirement is always satisfied by the configure-time
 *   determination of the Fortran parameter MPI_STATUS_SIZE.
 *
 * - A Fortran INTEGER should be the same size as a C int so
 *   that components indicated by MPI_SOURCE, MPI_TAG, and
 *   MPI_ERROR can be addressed properly from either language.
 *
 * - A Fortran status must be aligned such that all C status
 *   struct components have proper alignment.  The Fortran
 *   status alignment is only guaranteed to be suitable for
 *   Fortran INTEGERs.  The C status requires alignment for a
 *   size_t component.  We utilize two tests:
 *
 *   - Check if Fortran INTEGER alignment matches size_t alignment.
 *     This check is not necessary, but it is sufficient and can be
 *     assessed at compile time.
 *
 *   - Check if the particular Fortran status pointer provided by
 *     the user has suitable alignment.  This check is both necessary
 *     and sufficient, but must be conducted at run time.
 *
 *   These alignment issues are taken into consideration only
 *   for 64-bit SPARC runs, which is where these issues have
 *   been visible.
 */


/*
 * First, we have two preliminary checks:
 * - OMPI_FORTRAN_STATUS_NEED_CONVERSION_1
 *     is sufficient, but not necessary
 *     can be evaluated at compile time
 * - OMPI_FORTRAN_STATUS_NEED_CONVERSION_2(status)
 *     is sufficient and necessary
 *     must be evaluated at run time
 * If check #1 is false at compile time, then check #2 will always be false at run time.
 * The compile-time check is used to conditionalize compilation of references to c_status2.
 */


#if defined(__sparc) && SIZEOF_SIZE_T == 8
#define OMPI_FORTRAN_STATUS_NEED_CONVERSION_1 \
  ((OMPI_SIZEOF_FORTRAN_INTEGER!=SIZEOF_INT) || \
   (OMPI_ALIGNMENT_FORTRAN_INTEGER!=OPAL_ALIGNMENT_SIZE_T))
#else
#define OMPI_FORTRAN_STATUS_NEED_CONVERSION_1 \
   (OMPI_SIZEOF_FORTRAN_INTEGER!=SIZEOF_INT)
#endif


#if defined(__sparc) && SIZEOF_SIZE_T == 8
#define OMPI_FORTRAN_STATUS_NEED_CONVERSION_2(status) \
  ( \
    (OMPI_SIZEOF_FORTRAN_INTEGER!=SIZEOF_INT) \
    || \
    ( \
      (OMPI_ALIGNMENT_FORTRAN_INTEGER!=OPAL_ALIGNMENT_SIZE_T) \
      && \
      (((ulong) (status)) & (OPAL_ALIGNMENT_SIZE_T-1)) \
    ) \
  )
#else
#define OMPI_FORTRAN_STATUS_NEED_CONVERSION_2(status) \
   (OMPI_SIZEOF_FORTRAN_INTEGER!=SIZEOF_INT)
#endif


/*
 * Now, the macros:
 * - OMPI_FORTRAN_STATUS_DECLARATION(c_status,c_status2)
 * - OMPI_FORTRAN_STATUS_SET_POINTER(c_status,c_status2,status)
 * - OMPI_FORTRAN_STATUS_RETURN(c_status,c_status2,status,c_ierr)
 */


#if OMPI_FORTRAN_STATUS_NEED_CONVERSION_1
#define OMPI_FORTRAN_STATUS_DECLARATION(c_status,c_status2) MPI_Status *c_status, c_status2;
#else
#define OMPI_FORTRAN_STATUS_DECLARATION(c_status,c_status2) MPI_Status *c_status;
#endif


#if OMPI_FORTRAN_STATUS_NEED_CONVERSION_1
#define OMPI_FORTRAN_STATUS_SET_POINTER(c_status,c_status2,status) \
  do { \
      if (OMPI_IS_FORTRAN_STATUS_IGNORE(status)) { \
          c_status = MPI_STATUS_IGNORE; \
      } else { \
          if ( OMPI_FORTRAN_STATUS_NEED_CONVERSION_2(status) ) { \
              c_status = &c_status2; \
          } else { \
              c_status = (MPI_Status *) status; \
          } \
      } \
  } while (0);
#else
#define OMPI_FORTRAN_STATUS_SET_POINTER(c_status,c_status2,status) \
  do { \
      if (OMPI_IS_FORTRAN_STATUS_IGNORE(status)) { \
          c_status = MPI_STATUS_IGNORE; \
      } else { \
          c_status = (MPI_Status *) status; \
      } \
  } while (0);
#endif


#if OMPI_FORTRAN_STATUS_NEED_CONVERSION_1
#define OMPI_FORTRAN_STATUS_RETURN(c_status,c_status2,status,c_ierr) \
  do { \
      if ( \
          OMPI_FORTRAN_STATUS_NEED_CONVERSION_2(status) && \
          MPI_SUCCESS == c_ierr && \
          MPI_STATUS_IGNORE != c_status ) \
      { \
          MPI_Status_c2f(c_status, status); \
      } \
  } while (0);
#else
#define OMPI_FORTRAN_STATUS_RETURN(c_status,c_status2,status,c_ierr)
#endif


#endif /* OMPI_FORTRAN_STATUS_CONVERSION_H */
