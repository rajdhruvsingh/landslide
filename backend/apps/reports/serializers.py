from rest_framework import serializers
from .models import FieldReport


class FieldReportSerializer(serializers.ModelSerializer):
    user_phone = serializers.CharField(source="user.phone_number", read_only=True)

    class Meta:
        model = FieldReport
        fields = "__all__"
        read_only_fields = ["submitted_at"]


class FieldReportListSerializer(serializers.ModelSerializer):
    user_phone = serializers.CharField(source="user.phone_number", read_only=True)

    class Meta:
        model = FieldReport
        fields = (
            "id",
            "user",
            "user_phone",
            "geom",
            "report_type",
            "submitted_at",
            "sync_status",
        )
