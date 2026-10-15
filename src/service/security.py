"""Who is calling: the admin (holds AERO_ADMIN_TOKEN) or the public.  OWNER: Pranoy.

Only endpoints that spend money ask. Read-only endpoints stay open so the
hosted demo works in a logged-out browser.
"""

from __future__ import annotations

import hmac

from .settings import Settings

ADMIN, PUBLIC = "admin", "public"


class AuthError(Exception):
    """A credential was presented and is wrong, or public access is closed."""


def caller_role(authorization: str | None, settings: Settings) -> str:
    """Map an Authorization header to a role.

    No header means the public, if public runs are enabled. A header that is
    present but wrong is always refused, never silently treated as public, so
    a mistyped token is noticed. The comparison is constant-time.
    """
    if authorization is None or not authorization.strip():
        if settings.public_runs:
            return PUBLIC
        raise AuthError("this endpoint needs an admin token")
    scheme, _, token = authorization.strip().partition(" ")
    expected = settings.admin_token
    if (scheme.lower() == "bearer" and expected is not None
            and hmac.compare_digest(token.strip().encode(), expected.encode())):
        return ADMIN
    raise AuthError("invalid token")
