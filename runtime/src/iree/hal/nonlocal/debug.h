#ifndef DEBUG_PRINTF
#if defined(DEBUG) && DEBUG == 0
#undef DEBUG
#endif
#ifdef DEBUG
#include <stdio.h>
#define DEBUG_PRINTF(format, ...) printf(format, __VA_ARGS__)
#else
#define DEBUG_PRINTF(format, ...)
#endif
#endif
