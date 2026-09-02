from rest_framework import serializers
from .models import Alert


class AlertSerializer(serializers.ModelSerializer):
    zone_name = serializers.CharField(source="zone.zone_name", read_only=True)

    class Meta:
        model = Alert
        fields = "__all__"
        read_only_fields = ["dispatched_at"]
