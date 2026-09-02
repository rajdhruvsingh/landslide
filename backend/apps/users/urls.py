from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, login_view

router = DefaultRouter()
router.register(r"users", UserViewSet, basename="user")

urlpatterns = [
    path("auth/login", login_view, name="auth-login"),
    path("", include(router.urls)),
]
