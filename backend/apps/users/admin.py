from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ("phone_number", "role", "district", "preferred_language")
    list_filter = ("role", "district")
    fieldsets = BaseUserAdmin.fieldsets + (
        ("Landslide EWS", {"fields": ("phone_number", "role", "district", "preferred_language")}),
    )
