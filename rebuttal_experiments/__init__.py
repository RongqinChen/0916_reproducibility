"""Discussion-stage experiments, isolated from the submitted implementation.

Importing this package never launches experiments or downloads data.
"""

PROTOCOL_VERSION = "rebuttal-2026-11-12-v2"

if not __debug__:
    raise RuntimeError('Run without -O: scientific consistency checks must remain enabled.')
