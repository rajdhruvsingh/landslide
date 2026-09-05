from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.contrib.auth import get_user_model

from drf_spectacular.utils import extend_schema

from apps.users.permissions import IsAdmin
from .serializers import UserSerializer, LoginSerializer, TokenSerializer

User = get_user_model()


class UserViewSet(viewsets.ModelViewSet):
    """User management is admin-only; regular sign-up happens via /auth/login."""

    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdmin]


@extend_schema(
    request=LoginSerializer,
    responses={200: TokenSerializer, 400: None},
)
@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    """Phone + OTP login. Returns a JWT on success.

    ======================================================================
    DEV-ONLY STUB — DO NOT USE IN PRODUCTION
    ======================================================================
    No real OTP is generated or sent. This view accepts the fixed
    development codes "000000" or "test-otp" for ANY phone number and
    auto-creates a user for that phone if none exists.

    TODO(SMS): dispatch a real one-time password through the SMS gateway
    (settings.SMS_GATEWAY_*, currently mocked) and validate the submitted
    code server-side instead of comparing against hardcoded constants.
    ======================================================================
    """
    serializer = LoginSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    phone = serializer.validated_data["phone_number"]
    otp = serializer.validated_data["otp"]

    if otp != "000000" and otp != "test-otp":
        return Response(
            {"error": "Invalid OTP"},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    user, _created = User.objects.get_or_create(
        phone_number=phone,
        defaults={"username": phone},
    )

    from apps.users.token_utils import create_token

    token = create_token(user)

    return Response(
        {"access_token": token, "token_type": "bearer"},
        status=status.HTTP_200_OK,
    )
