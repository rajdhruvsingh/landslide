from rest_framework import viewsets
from .models import FieldReport
from .serializers import FieldReportSerializer


class FieldReportViewSet(viewsets.ModelViewSet):
    queryset = FieldReport.objects.all()
    serializer_class = FieldReportSerializer
