#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A proxy server enabling GCE VMs in a Managed Instance Group to delete themselves.

GCE Managed instance groups don't have any good way to handle autoscaling for
long-running workloads. With the autoscaler configured to scale in, instances
get only 90 seconds warning to shut down. So we set the autoscaler to only scale
out and have the VMs tear themselves down when they're down with their work. But
anything that brings down the VM other than a delete call to the instance group
manager API is considered as an "unhealthy" VM, which gets created in exactly
the same configuration, regardless of any update or autoscaling settings. Making
the correct API call requires broad permissions on the instance group manager,
which we don't want to give the VMs. To scope permissions to individual
instances, this proxy service makes use of instance identity tokens to allow an
instance to make a call only to delete itself.

See https://cloud.google.com/compute/docs/instances/verifying-instance-identity

Note that http.server is not in general fit for production, but our very limited
usage of BaseHTTPRequestHandler here, not serving any files and not doing our
own parsing of request input, does not present any security concerns. Don't add
those sorts of things.
"""

import argparse
import http.server

import google.auth.transport.requests
import requests
from google.cloud import compute
from google.oauth2 import id_token

AUTH_HEADER_PREFIX = "Bearer "
# TODO: Inject appropriate audience.
AUDIENCE = "localhost"


def verify_token(token: str, audience: str) -> dict:
  """Verify token signature and return the token payload"""
  request = google.auth.transport.requests.Request()
  # TODO: Validate that parsing here is safe.
  payload = id_token.verify_token(token, request=request, audience=audience)
  return payload


class InstanceDeleteHandler(http.server.BaseHTTPRequestHandler):

  def do_DELETE(self):
    try:
      # Since the auth token contains all the information we need, no path is
      # needed.
      if self.path != "/":
        return self.send_error(400, "Invalid path (only root path is valid)")

      auth_header = self.headers.get("Authorization")
      if auth_header is None or not auth_header.startswith(AUTH_HEADER_PREFIX):
        return self.send_error(401, "Missing or malformed authorization header")

      token = auth_header[len(AUTH_HEADER_PREFIX):]

      try:
        token_payload = verify_token(token, AUDIENCE)["google"]["compute_engine"]
      except Exception as e:
        # Consider giving the client more information?
        self.log_error("%s", e)
        return self.send_error(401, "Bearer token is malformed")

      # We now have a payload identifying the VM that made the request to delete
      # itself.
      self.log_message("Token payload: %s", token_payload)
      # TODO: create a server class and persist clients.
      instances_client = compute.InstancesClient()
      migs_client = compute.RegionInstanceGroupManagersClient()

      project = token_payload["project_id"]
      instance = instances_client.get(instance=token_payload["instance_name"],
                                      project=project,
                                      zone=token_payload["zone"])
      # Verify it's *actually* the same instance. Names get reused, but IDs
      # don't. For some reason you can't reference instances by their ID in any
      # of the APIs.
      if instance.id != int(token_payload["instance_id"]):
        return self.send_error(
            400, "Existing instance of the same name has a different ID.")

      # Why would the Python API return something a silly as a dictionary?
      mig = next((item.value
                  for item in instance.metadata.items
                  if item.key == "created-by"), None)
      if mig is None:
        return self.send_error(
            400, "Instance is not part of a managed instance group.")

      # Drop the trailing zone identifier to get the region. Yeah it kinda does
      # seem like there should be a better way to do this...
      region, _ = instance.zone.rsplit("-", maxsplit=1)
      # TODO: correctly handle and forward (if appropriate) errors from the API
      migs_client.delete_instances(
          instance_group_manager=mig,
          project=project,
          region=region,
          # For some reason we can't just use a list of instance names...
          region_instance_group_managers_delete_instances_request_resource=
          compute.RegionInstanceGroupManagersDeleteInstancesRequest(
              instances=[instance.self_link]))

      self.send_response(200)
      self.send_header("Content-type", "text/html")
      self.end_headers()
    # If anything else fails, send the user a generic internal server error response.
    except Exception as e:
      self.log_error("%s", e)
      return self.send_error(500)


def main(args):
  webServer = http.server.HTTPServer(("", args.port), InstanceDeleteHandler)
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
