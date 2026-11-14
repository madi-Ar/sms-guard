"""HTML-страницы используют существующую сессию и REST API."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST
from django.contrib import messages
from django.core.paginator import Paginator
from django.db import transaction
from .models import Check
from .serializers import HistoryFilters


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


@login_required(login_url='/login/')
@require_GET
def history_page(request):
    values = {k: v for k, v in request.GET.items() if v and k != 'page'}
    filters = HistoryFilters(data=values)
    query = Check.objects.filter(user=request.user)
    valid = filters.is_valid()
    if valid:
        for key, value in filters.validated_data.items():
            lookup = {'date_from': 'created_at__date__gte', 'date_to': 'created_at__date__lte'}.get(key, key)
            query = query.filter(**{lookup: value})
    else:
        query = query.none()
    params = request.GET.copy()
    params.pop('page', None)
    page = Paginator(query, 20).get_page(request.GET.get('page'))
    return render(request, 'analysis/history.html', {
        'page_obj': page, 'filter_query': params.urlencode(),
        'filter_errors': filters.errors, 'categories': Check.CATEGORIES,
        'levels': Check.LEVELS, 'types': Check.TYPES}, status=200 if valid else 400)


@login_required(login_url='/login/')
@require_POST
def delete_check(request, pk):
    check = get_object_or_404(Check, pk=pk, user=request.user)
    if request.POST.get('confirm') != 'yes':
        messages.error(request, 'Подтвердите удаление проверки.')
        return redirect('check-result', pk=pk)
    with transaction.atomic():
        check.delete()
    messages.success(request, 'Проверка удалена.')
    return redirect('history')
