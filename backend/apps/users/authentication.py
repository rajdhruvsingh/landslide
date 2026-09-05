from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions

from .token_utils import decode_token

User = get_user_model()


class JWTAuthentication(authentication.BaseAuthentication):
    """DRF authentication backend that reads a Bearer JWT.

    Tokens are produced by apps.users.views.login_view and validated via
    apps.users.token_utils.decode_token.
    """

    keyword = "Bearer"

    def authenticate(self, request):
        header = authentication.get_authorization_header(request).split()
        if not header or header[0].lower() != self.keyword.lower().encode():
            return None
        if len(header) != 2:
            raise exceptions.AuthenticationFailed(
                "Invalid Authorization header. Expected: Authorization: Bearer <token>"
            )
        user = decode_token(header[1].decode())
        if user is None:
            raise exceptions.AuthenticationFailed("Invalid or expired token")
        return (user, header[1].decode())

    def authenticate_header(self, request):
        return self.keyword