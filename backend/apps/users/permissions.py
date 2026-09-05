from rest_framework.permissions import BasePermission

# Product-role mapping (User.role choices):
#   citizen, field_official, district_admin, state_admin, system
# "admin / district officer" (staff)  = field_official + district_admin + state_admin
# "admin" (write access to risk data) = district_admin + state_admin
ADMIN_ROLES = {"district_admin", "state_admin"}
STAFF_ROLES = ADMIN_ROLES | {"field_official"}


def _has_role(user, roles):
    return bool(
        user
        and user.is_authenticated
        and (user.is_staff or user.is_superuser or user.role in roles)
    )


class IsAdmin(BasePermission):
    """Allow only district/state admins to write risk-zone data.

    Django staff/superusers are always allowed as an escape hatch.
    The ML pipeline/Celery writes risk zones directly through the ORM,
    not via the API, so this only guards manual changes.
    """

    def has_permission(self, request, view):
        return _has_role(request.user, ADMIN_ROLES)


class IsStaffOrAdmin(BasePermission):
    """Allow admins, district admins, and field officials (deny citizens).

    Applies the product rule: "admin/district officer only" for alert
    dispatch and report management.
    """

    def has_permission(self, request, view):
        return _has_role(request.user, STAFF_ROLES)