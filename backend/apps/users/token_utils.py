from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

User = get_user_model()


def create_token(user):
    """Create a JWT-like token for a user.

    Uses python-jose if available, otherwise a simple signed token.
    """
    try:
        from datetime import timedelta

        from jose import jwt

        expire = timezone.now() + timedelta(
            minutes=getattr(settings, "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60)
        )
        payload = {
            "sub": str(user.pk),
            "phone": user.phone_number,
            "role": user.role,
            "exp": expire.timestamp(),
        }
        return jwt.encode(
            payload,
            getattr(settings, "SECRET_KEY", "dev-secret"),
            algorithm="HS256",
        )
    except ImportError:
        import hashlib
        import hmac

        raw = f"{user.pk}:{user.phone_number}:{timezone.now().timestamp()}"
        sig = hmac.new(
            settings.SECRET_KEY.encode(), raw.encode(), hashlib.sha256
        ).hexdigest()
        return f"{user.pk}.{sig}"


def decode_token(token):
    """Decode and validate a token. Returns the user or None."""
    try:
        from jose import jwt as jose_jwt

        payload = jose_jwt.decode(
            token,
            getattr(settings, "SECRET_KEY", "dev-secret"),
            algorithms=["HS256"],
        )
        return User.objects.get(pk=int(payload["sub"]))
    except Exception:
        return None
