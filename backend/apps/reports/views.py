from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from apps.users.permissions import IsStaffOrAdmin
from .models import FieldReport
from .serializers import FieldReportSerializer, FieldReportListSerializer


class FieldReportViewSet(viewsets.ModelViewSet):
    queryset = FieldReport.objects.select_related("user").all()
    serializer_class = FieldReportSerializer

    def get_permissions(self):
        if self.action == "create":
            # Any authenticated user (citizen, field official, admin) may report.
            return [IsAuthenticated()]
        # Listing, updating, and deleting reports are admin/district-officer only.
        return [IsStaffOrAdmin()]

    def get_serializer_class(self):
        if self.action == "list":
            return FieldReportListSerializer
        return FieldReportSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def get_queryset(self):
        qs = super().get_queryset()
        zone_id = self.request.query_params.get("zone_id")
        sync_status = self.request.query_params.get("status")
        if zone_id is not None:
            qs = qs.filter(geom__isnull=False)
        if sync_status:
            qs = qs.filter(sync_status=sync_status)
        return qs