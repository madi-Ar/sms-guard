from django.urls import path
from .ui import check_page, result_page

urlpatterns = [
    path('check/sms/', check_page, {'kind': 'sms'}, name='check-sms'),
    path('check/url/', check_page, {'kind': 'url'}, name='check-url'),
    path('checks/<int:pk>/', result_page, name='check-result'),
]
