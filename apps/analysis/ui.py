"""HTML-страницы используют существующую сессию и REST API."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .models import Check


@login_required(login_url='/login/')
@require_GET
def check_page(request, kind):
    return render(request, 'analysis/check.html', {'kind': kind})


@login_required(login_url='/login/')
@require_GET
def result_page(request, pk):
    check = Check.objects.filter(pk=pk, user=request.user).first()
    if check is None:
        return render(request, 'analysis/not_found.html', status=404)
    return render(request, 'analysis/result.html', {'check': check})
