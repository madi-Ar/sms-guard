from django import forms
from .models import UserProfile


class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['preferred_language', 'store_sms_text']
        labels = {'preferred_language': 'Предпочтительный язык',
                  'store_sms_text': 'Сохранять полный текст новых SMS'}
