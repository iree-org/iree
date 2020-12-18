// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "experimental/remoting/iree/remoting/support/io_loop.h"

#include "experimental/remoting/iree/remoting/support/linux_uring.h"
#include "iree/base/logging.h"

#if IREE_REMOTING_HAVE_URING
#include "experimental/remoting/iree/remoting/support/io_loop_uring.cc.inc"
#endif

namespace iree {
namespace remoting {

//------------------------------------------------------------------------------
// IoLoop
//------------------------------------------------------------------------------

iree_status_t IoLoop::Create(std::unique_ptr<IoLoop> &created) {
#if IREE_REMOTING_HAVE_URING
  {
    iree_status_t status = UringImpl::TryCreateSpecific(created);
    if (iree_status_is_ok(status)) {
      IREE_DVLOG(1) << "Created uring IoLoop";
      return status;
    }
    iree_status_ignore(status);
  }

  IREE_DLOG(WARNING) << "No compatible IoLoop provider found";
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
#endif
}

void IoLoop::SubmitGeneric(IoRequest *request) {
  // Note that most real systems will have a single implementation (or at
  // most a couple), and they do not vary for the lifetime. We therefore
  // probe them in priority order, avoiding an indirect dispatch.
  auto t = impl_type_;
#if IREE_REMOTING_HAVE_URING
  if (t == ImplType::kUring) {
    static_cast<UringImpl *>(this)->SubmitSpecific(request);
    return;
  }
#endif

  IREE_CHECK(false) << "SubmitGeneric unmatched ImplType: "
                    << static_cast<int>(t);
}

}  // namespace remoting
}  // namespace iree
