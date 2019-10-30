#ifndef SDL_config_h_
#define SDL_config_h_

#include "SDL_platform.h"

/**
 *  \file SDL_config.h
 */

/* Add any platform that doesn't build using the configure system. */
#if defined(__WIN32__)
#include "SDL_config_windows.h"
#elif defined(__MACOSX__)
#include "SDL_config_macosx.h"
#elif defined(__IPHONEOS__)
#include "SDL_config_iphoneos.h"
#elif defined(__ANDROID__)
#include "SDL_config_android.h"
#elif defined(__LINUX__)
#include "SDL_config_linux.h"
#elif defined(__EMSCRIPTEN__)
#include "SDL_config_emscripten.h"
#else
/* This is a minimal configuration just to get SDL running on new platforms */
#include "SDL_config_minimal.h"
#endif /* platform config */

#ifdef USING_GENERATED_CONFIG_H
#error Wrong SDL_config.h, check your include path?
#endif

#endif /* SDL_config_h_ */
