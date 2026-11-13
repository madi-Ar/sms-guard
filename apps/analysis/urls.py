from django.urls import path
from apps.accounts.views import LoginView, LogoutView, RegisterView
from .views import CheckDetailView, CheckView, HealthView, HistoryView, SummaryView, URLCheckView

urlpatterns = [
    path('health/', HealthView.as_view(), name='api-health'),
    path('auth/register/', RegisterView.as_view(), name='api-register'),
    path('auth/login/', LoginView.as_view(), name='api-login'),
    path('auth/logout/', LogoutView.as_view(), name='api-logout'),
    path('check/sms/', CheckView.as_view(), name='api-check-sms'),
    path('check/url/', URLCheckView.as_view(), name='api-check-url'),
    path('checks/', HistoryView.as_view(), name='api-checks'),
    path('checks/<int:pk>/', CheckDetailView.as_view(), name='api-check-detail'),
    path('stats/summary/', SummaryView.as_view(), name='api-summary'),
]
