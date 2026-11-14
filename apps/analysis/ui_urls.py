from django.urls import path
from .ui import check_page, result_page, history_page, delete_check

urlpatterns = [
    path('history/', history_page, name='history'),
    path('checks/<int:pk>/delete/', delete_check, name='check-delete'),
    path('check/sms/', check_page, {'kind': 'sms'}, name='check-sms'),
    path('check/url/', check_page, {'kind': 'url'}, name='check-url'),
    path('checks/<int:pk>/', result_page, name='check-result'),
]
