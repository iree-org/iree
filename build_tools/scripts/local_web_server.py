#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Local server for development, with support for CORS headers and MIME types.

NOTE: This is NOT suitable for production serving, it is just a slightly
extended version of https://docs.python.org/3/library/http.server.html.

Usage:
  python3 local_server.py --directory {build_dir}
  (then open http://localhost:8000/ in your browser)
"""

import os
from functools import partial
from http import server


class CORSHTTPRequestHandler(server.SimpleHTTPRequestHandler):

  def __init__(self, *args, **kwargs):
    # Include MIME types for files we expect to be serving.
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    self.extensions_map.update({
        ".js": "application/javascript",
        ".wasm": "application/wasm",
    })
    super().__init__(*args, **kwargs)

  # Inspiration for this hack: https://stackoverflow.com/a/13354482
  def end_headers(self):
    self.send_cors_headers()

    server.SimpleHTTPRequestHandler.end_headers(self)

  def send_cors_headers(self):
    # Emscripten uses SharedArrayBuffer for its multithreading, which requires
    # Cross Origin Opener Policy and Cross Origin Embedder Policy headers:
    #   * https://emscripten.org/docs/porting/pthreads.html
    #   * https://developer.chrome.com/blog/enabling-shared-array-buffer/
    self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
    self.send_header("Cross-Origin-Opener-Policy", "same-origin")


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--directory',
                      '-d',
                      default=os.getcwd(),
                      help='Specify alternative directory '
                      '[default:current directory]')
  parser.add_argument('port',
                      action='store',
                      default=8000,
                      type=int,
                      nargs='?',
                      help='Specify alternate port [default: 8000]')
  args = parser.parse_args()

  server.test(HandlerClass=partial(CORSHTTPRequestHandler,
                                   directory=args.directory),
              port=args.port)
