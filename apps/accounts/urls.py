from django.urls import path
from .views import auth_page

urlpatterns = [
    path('login/', auth_page, {'mode': 'login'}, name='login'),
    path('register/', auth_page, {'mode': 'register'}, name='register'),
]
