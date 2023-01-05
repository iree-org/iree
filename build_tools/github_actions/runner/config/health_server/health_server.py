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
import glob
import http.server
import subprocess
from http.client import INTERNAL_SERVER_ERROR, NOT_FOUND, OK
from typing import Optional

RUNNER_SERVICE_NAME = "gh-runner"
CHECK_SERVICE_CMD = ["systemctl", "is-active", RUNNER_SERVICE_NAME]
CHECK_SERVICE_TIMEOUT = 10
RUNNER_WORK_LOG_PATTERN = "/runner-root/actions-runner/_diag/Worker_*"


class HealthCheckHandler(http.server.BaseHTTPRequestHandler):

  def send_success(self,
                   *,
                   msg: Optional[str] = None,
                   body: Optional[str] = None):
    self.send_response(OK)
    self.send_header("Content-type", "text/html")
    self.end_headers()
    if body is not None:
      self.wfile.write(bytes(body, encoding="utf-8"))

  def do_GET(self):
    try:
      subprocess.run(CHECK_SERVICE_CMD,
                     check=True,
                     text=True,
                     stdout=subprocess.PIPE,
                     timeout=CHECK_SERVICE_TIMEOUT)
    except subprocess.TimeoutExpired as e:
      msg = f"'{' '.join(e.cmd)}' timed out: {e.stdout}"
      return self.send_error(INTERNAL_SERVER_ERROR, msg)
    except subprocess.CalledProcessError as e:
      return self.send_error(
          NOT_FOUND, f"Runner service not found: '{' '.join(e.cmd)}' returned"
          f" '{e.stdout.strip()}' (exit code {e.returncode})")

    # The runner writes a log file for each job it runs. In our case it only
    # runs one, so we glob for anything matching that pattern. Yes that is an
    # absolutely ludicrous way to get the runner's status. GitHub should really
    # implement a proper health check so we don't have to hack around like this.
    if glob.glob(RUNNER_WORK_LOG_PATTERN):
      return self.send_success(body="active")

    return self.send_success(body="idle")


def main(args: argparse.Namespace):
  webServer = http.server.HTTPServer(("", args.port), HealthCheckHandler)
  print(f"Server started on port {args.port}. Ctrl+C to stop.")

  try:
    webServer.serve_forever()
  except KeyboardInterrupt:
    # Don't print an exception on interrupt. Add a newline to handle printing of
    # "^C"
    print()

  webServer.server_close()
  print("Server stopped.")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", type=int, default=8080)
  return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
