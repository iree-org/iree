# IREE Remoting Protocol

## Version 1

The IREE remoting protocol is a binary, packet based, bi-directional, stream
based protocol.

When a new communication channel is established, both sides send a *handshake*
message, which is a fixed-length binary struct (see
`iree_remoting_handshake_v1_wire_t`) with parameters for setting up the transport
(note that this phase can be optional in certain embedded situations where both
ends can be configured with a pre-set handshake).
