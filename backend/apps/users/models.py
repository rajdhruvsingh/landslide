from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    ROLES = [
        ("citizen", "Citizen"),
        ("field_official", "Field Official"),
        ("district_admin", "District Admin"),
        ("state_admin", "State Admin"),
        ("system", "System/ML Pipeline"),
    ]

    phone_number = models.CharField(max_length=20, unique=True)
    role = models.CharField(max_length=20, choices=ROLES, default="citizen")
    district = models.CharField(max_length=100, blank=True, default="")
    preferred_language = models.CharField(max_length=10, default="en")

    class Meta:
        db_table = "users"

    def __str__(self):
        return f"{self.phone_number} ({self.role})"
