from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from apps.dashboard import dashboard_summary
import apps.users.schemas  # noqa: F401  (registers the JWTAuth OpenAPI scheme)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/docs/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
    path("api/v1/dashboard/summary", dashboard_summary, name="dashboard-summary"),
    path("api/v1/", include("apps.risk_zones.urls")),
    path("api/v1/", include("apps.reports.urls")),
    path("api/v1/", include("apps.alerts.urls")),
    path("api/v1/", include("apps.users.urls")),
    path("api/v1/", include("apps.weather.urls")),
]