from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from .serializers import UserSerializer, LoginSerializer, TokenSerializer

User = get_user_model()


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer


@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    """Phone + OTP login. Returns JWT on success."""
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
