from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("apps.risk_zones.urls")),
    path("api/v1/", include("apps.reports.urls")),
    path("api/v1/", include("apps.alerts.urls")),
    path("api/v1/", include("apps.users.urls")),
    path("api/v1/", include("apps.weather.urls")),
]
