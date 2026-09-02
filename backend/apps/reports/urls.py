from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import FieldReportViewSet

router = DefaultRouter()
router.register(r"reports", FieldReportViewSet)

urlpatterns = [
    path("", include(router.urls)),
]
