#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A really basic health check HTTP server.

All it does is server a 200 to every request, basically only confirming its
existence. This can later be extended with more functionality.

Note that http.server is not in general fit for production, but our very limited
usage of BaseHTTPRequestHandler here, not serving any files and not parsing or
making use of request input, does not present any security concerns. Don't add
those sorts of things. Additionally, this operates inside a firewall that blocks
all but a few IPs and even those are internal to the network.
"""

import argparse
import http.server


class HealthCheckHandler(http.server.BaseHTTPRequestHandler):

  def do_GET(self):
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()


def main(args):
  webServer = http.server.HTTPServer(("", args.port), HealthCheckHandler)
  print(f"Server started on port {args.port}. Ctrl+C to stop.")

  try:
    webServer.serve_forever()
  except KeyboardInterrupt:
    # Don't print an exception on interrupt. Add a newline to handle printing of
    # "^C"
    print()
    pass

  webServer.server_close()
  print("Server stopped.")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", type=int, default=8080)
  return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
