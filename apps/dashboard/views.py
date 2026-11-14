from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET
from rest_framework.exceptions import ValidationError
from .statistics import parse_period


@login_required(login_url='/login/')
@require_GET
def index(request):
    try:
        period = parse_period(request.GET.get('period', '30'))
    except ValidationError:
        return render(request, 'dashboard/index.html', {'period_error': True}, status=400)
    return render(request, 'dashboard/index.html', {'period': period})
