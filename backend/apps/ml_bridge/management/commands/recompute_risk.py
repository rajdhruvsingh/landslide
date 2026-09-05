"""Recompute baseline risk levels for all monitored risk zones.

Deterministic, offline, threshold-model driven. See apps.ml_bridge.risk_evaluator.
Example:
    python manage.py recompute_risk
    python manage.py recompute_risk --zone 1 6
"""

import json

from django.core.management.base import BaseCommand

from apps.ml_bridge.risk_evaluator import recompute_zone_risks


class Command(BaseCommand):
    help = (
        "Recompute baseline risk levels + last_computed_at from the latest "
        "weather readings using the published threshold model."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--zone",
            type=int,
            nargs="*",
            dest="zone_ids",
            help="Restrict recomputation to these RiskZone primary keys.",
        )

    def handle(self, *args, **options):
        result = recompute_zone_risks(zone_ids=options.get("zone_ids"))
        self.stdout.write(json.dumps(result, indent=2, default=str))