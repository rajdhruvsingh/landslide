"""Seed the complete demo dataset for the live E2E judging script.

Idempotent. Creates/refreshes:
  * state_admin user  (phone 9876543210, OTP login works via test-otp)
  * 5 base NER risk zones with realistic-risk levels + 2 alerts + 2 field reports
  * the Kalimpong (West Bengal) test zone at baseline ``Low`` risk
  * one recent AWS rainfall reading of 87.6 mm for the test zone — high enough
    to exceed the published NE-Himalaya threshold over the operational 48 h
    window (E = -11.10 + 0.62*D = 0.62*48 - 11.10 = 18.66 mm), which drives the
    recompute -> High + alert flow performed in the demo.

Example:
    python manage.py demo_seed
"""

import json

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta

from django.contrib.auth import get_user_model

from apps.alerts.models import Alert
from apps.reports.models import FieldReport
from apps.risk_zones.models import RiskZone
from apps.weather.models import WeatherReading

User = get_user_model()

BASE_ZONES = [
    {
        "zone_name": "Shillong Urban Cluster",
        "district": "East Khasi Hills",
        "state": "Meghalaya",
        "risk": "Severe",
        "geom": {
            "type": "Polygon",
            "coordinates": [
                [
                    [91.80, 25.50],
                    [91.98, 25.50],
                    [91.98, 25.66],
                    [91.80, 25.66],
                    [91.80, 25.50],
                ]
            ],
        },
    },
    {
        "zone_name": "Guwahati Metropolitan",
        "district": "Kamrup Metropolitan",
        "state": "Assam",
        "risk": "High",
        "geom": {
            "type": "Polygon",
            "coordinates": [
                [
                    [91.62, 26.08],
                    [91.86, 26.08],
                    [91.86, 26.22],
                    [91.62, 26.22],
                    [91.62, 26.08],
                ]
            ],
        },
    },
    {
        "zone_name": "Imphal Valley",
        "district": "Imphal West",
        "state": "Manipur",
        "risk": "Moderate",
        "geom": {
            "type": "Polygon",
            "coordinates": [
                [
                    [93.86, 24.76],
                    [94.02, 24.76],
                    [94.02, 24.86],
                    [93.86, 24.86],
                    [93.86, 24.76],
                ]
            ],
        },
    },
    {
        "zone_name": "Kohima Town",
        "district": "Kohima",
        "state": "Nagaland",
        "risk": "Low",
        "geom": {
            "type": "Polygon",
            "coordinates": [
                [
                    [94.06, 25.64],
                    [94.14, 25.64],
                    [94.14, 25.72],
                    [94.06, 25.72],
                    [94.06, 25.64],
                ]
            ],
        },
    },
    {
        "zone_name": "Lunglei District",
        "district": "Lunglei",
        "state": "Mizoram",
        "risk": "Moderate",
        "geom": {
            "type": "Polygon",
            "coordinates": [
                [
                    [92.62, 22.80],
                    [92.84, 22.80],
                    [92.84, 23.00],
                    [92.62, 23.00],
                    [92.62, 22.80],
                ]
            ],
        },
    },
]

TEST_ZONE = {
    "zone_name": "Kalimpong-Darjeeling Foothills",
    "district": "Kalimpong",
    "state": "West Bengal",
    "risk": "Low",  # baseline; demo recompute will elevate it to High
    "geom": {
        "type": "Polygon",
        "coordinates": [
            [
                [88.35, 26.98],
                [88.56, 26.98],
                [88.56, 27.12],
                [88.35, 27.12],
                [88.35, 26.98],
            ]
        ],
    },
}

TEST_RAINFALL_MM = 87.6
TEST_STATION_ID = "KLP-AWS-001"


class Command(BaseCommand):
    help = "Seed the complete, reproducible demo dataset for the E2E judging script (idempotent)."

    def handle(self, *args, **options):
        user, _ = User.objects.get_or_create(
            phone_number="9876543210",
            defaults={
                "username": "9876543210",
                "role": "state_admin",
                "is_staff": True,
            },
        )

        zones_by_name = {}
        now = timezone.now()
        for z in BASE_ZONES + [TEST_ZONE]:
            zone, _ = RiskZone.objects.update_or_create(
                zone_name=z["zone_name"],
                defaults={
                    "district": z["district"],
                    "state": z["state"],
                    "geom": z["geom"],
                    "current_risk_level": z["risk"],
                    "last_computed_at": None
                    if z is TEST_ZONE
                    else now - timedelta(minutes=11),
                },
            )
            zones_by_name[z["zone_name"]] = zone

        # Remove any legacy test-zone rows (e.g. from an older zone_name).
        RiskZone.objects.filter(zone_name__startswith="Kalimpong").exclude(
            zone_name=TEST_ZONE["zone_name"]
        ).delete()

        Alert.objects.update_or_create(
            zone=zones_by_name["Shillong Urban Cluster"],
            risk_level="Severe",
            defaults={
                "message": (
                    "Heavy rainfall past 24h; slope saturation approaching "
                    "failure threshold in Shillong urban cluster."
                ),
                "explanation": (
                    "Rainfall 128.4 mm in 24h exceeds the 100 mm alarm threshold "
                    "for NE Himalaya; soil moisture 68%."
                ),
                "channel": "both",
                "language": "en",
            },
        )
        Alert.objects.update_or_create(
            zone=zones_by_name["Guwahati Metropolitan"],
            risk_level="High",
            defaults={
                "message": (
                    "Sustained moderate rain is destabilising cut slopes along "
                    "NH-27 escarpments."
                ),
                "explanation": (
                    "Rainfall 74.2 mm in 24h with 3 consecutive rain days; "
                    "advisory-level threshold reached."
                ),
                "channel": "sms",
                "language": "en",
            },
        )

        FieldReport.objects.update_or_create(
            user=user,
            report_type="landslide",
            defaults={
                "description": (
                    "Fresh crack and debris on roadside slope near Police Bazaar, "
                    "Shillong."
                ),
                "sync_status": "pending",
                "photo_url": "https://picsum.photos/seed/landslide1/200/150",
            },
        )
        FieldReport.objects.update_or_create(
            user=user,
            report_type="road_block",
            defaults={
                "description": "Slip blocking single lane on NH-6, Umiam stretch.",
                "sync_status": "synced",
                "photo_url": "https://picsum.photos/seed/roadblock2/200/150",
            },
        )

        # Fresh demo reading for the test zone (replace any stale demo rows).
        test_zone = zones_by_name[TEST_ZONE["zone_name"]]
        WeatherReading.objects.filter(
            zone=test_zone,
            station_id=TEST_STATION_ID,
        ).delete()
        WeatherReading.objects.create(
            zone=test_zone,
            station_id=TEST_STATION_ID,
            reading_time=now - timedelta(minutes=30),
            rainfall_mm=TEST_RAINFALL_MM,
            soil_moisture_pct=68.0,
            source="AWS_DEMO",
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Demo dataset ready: "
                f"{RiskZone.objects.count()} zones | "
                f"{Alert.objects.count()} alerts | "
                f"{FieldReport.objects.count()} reports | "
                f"{WeatherReading.objects.count()} readings | "
                f"user {user.phone_number} ({user.role})\n"
                f"Test zone '{TEST_ZONE['zone_name']}' at baseline "
                f"{test_zone.current_risk_level} with reading "
                f"{TEST_RAINFALL_MM} mm (E_threshold_48h = 18.66 mm) -> "
                f"recompute_risk will elevate it to High."
            )
        )