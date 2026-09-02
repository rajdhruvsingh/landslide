from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import FieldReport
from .serializers import FieldReportSerializer, FieldReportListSerializer


class FieldReportViewSet(viewsets.ModelViewSet):
    queryset = FieldReport.objects.select_related("user").all()
    serializer_class = FieldReportSerializer

    def get_serializer_class(self):
        if self.action == "list":
            return FieldReportListSerializer
        return FieldReportSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        zone_id = self.request.query_params.get("zone_id")
        sync_status = self.request.query_params.get("status")
        if zone_id is not None:
            qs = qs.filter(geom__isnull=False)
        if sync_status:
            qs = qs.filter(sync_status=sync_status)
        return qs
