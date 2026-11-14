"""Маршруты проекта: главная страница и базовый health API."""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.dashboard.urls')),
    path('', include('apps.accounts.urls')),
    path('', include('apps.analysis.ui_urls')),
    path('api/v1/', include('apps.analysis.urls')),
]
