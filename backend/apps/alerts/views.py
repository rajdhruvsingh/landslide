from django.utils import timezone
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Alert
from .serializers import AlertSerializer


class AlertViewSet(viewsets.ModelViewSet):
    queryset = Alert.objects.select_related("zone").all()
    serializer_class = AlertSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        district = self.request.query_params.get("district")
        since = self.request.query_params.get("since")
        if district:
            qs = qs.filter(zone__district=district)
        if since:
            qs = qs.filter(dispatched_at__gte=since)
        return qs

    @action(detail=False, methods=["post"], url_path="dispatch")
    def dispatch_alert(self, request):
        """Manually trigger an alert (admin override)."""
        from apps.alerts.rules_engine import should_dispatch_alert

        zone_id = request.data.get("zone_id")
        risk_level = request.data.get("risk_level", "High")
        message = request.data.get("message", "")
        language = request.data.get("language", "en")
        channel = request.data.get("channel", "sms")
        explanation = request.data.get("explanation", "")

        if not zone_id:
            return Response(
                {"error": "zone_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = should_dispatch_alert(zone_id, risk_level, None)

        if not result["should_dispatch"]:
            return Response(
                {"status": "skipped", "reason": result["reason"]},
                status=status.HTTP_200_OK,
            )

        alert = Alert.objects.create(
            zone_id=zone_id,
            risk_level=risk_level,
            message=message,
            language=language,
            channel=channel,
            explanation=explanation,
        )
        return Response(
            {"status": "dispatched", "alert_id": alert.pk},
            status=status.HTTP_201_CREATED,
        )
