"""API поверх существующего анализа; история содержит маскированный ввод."""
import logging
from time import perf_counter

from django.db.models import Avg, Count, Q
from django.middleware.csrf import get_token
from django.shortcuts import get_object_or_404
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.predictor import ARTIFACTS
from . import services
from .api import CheckThrottle, ModelUnavailable
from .models import Check
from .serializers import HistoryFilters, SMSInput, URLInput
from .persistence import mask_input, private_result, save_check

logger = logging.getLogger(__name__)


class CheckView(APIView):
    throttle_classes = [CheckThrottle]
    check_type = 'sms'

    def post(self, request):
        started = perf_counter()
        check = None
        status = 201
        try:
            serializer = (SMSInput if self.check_type == 'sms' else URLInput)(data=request.data)
            serializer.is_valid(raise_exception=True)
            source = serializer.validated_data['text' if self.check_type == 'sms' else 'url']
            try:
                result = (services.analyze_sms(source) if self.check_type == 'sms'
                          else services.analyze_url(source)).to_dict()
            except (RuntimeError, OSError):
                raise ModelUnavailable() from None
            if self.check_type == 'url':
                result.update(type='url', p_sms=None, risk_score=None, risk_level=None)
            check = save_check(request.user, self.check_type, source, result)
            result.update(id=check.pk, created_at=check.created_at.isoformat())
            return Response(result, status=201)
        except Exception as exc:
            status = getattr(exc, 'status_code', 500)
            raise
        finally:
            logger.info('analysis duration_ms=%.2f check_id=%s category=%s status=%s',
                        (perf_counter() - started) * 1000,
                        check.pk if check else None, check.category if check else None, status)


class URLCheckView(CheckView):
    check_type = 'url'


def history_data(check):
    data = {**check.result_json, 'masked_input': check.masked_input}
    data.update(id=check.pk, type=check.type, category=check.category,
                created_at=check.created_at.isoformat())
    for name in ('language', 'risk_level', 'risk_score', 'p_sms', 'p_url', 'p_url_max',
                 'explanation', 'recommendation'):
        value = getattr(check, name)
        if value is not None:
            data[name] = value
    versions = dict(data.get('model_versions') or {})
    for kind in ('sms', 'url'):
        version = getattr(check, f'{kind}_model_version')
        if version is not None:
            versions[kind] = version
    if versions:
        data['model_versions'] = versions
    if check.input_text is not None:
        data['input_text'] = check.input_text
    if check.demo_key:
        data['is_demo'] = True
    return data


class HistoryView(APIView):
    def get(self, request):
        filters = HistoryFilters(data=request.query_params)
        filters.is_valid(raise_exception=True)
        query = Check.objects.filter(user=request.user)
        for key, value in filters.validated_data.items():
            lookup = {'date_from': 'created_at__date__gte',
                      'date_to': 'created_at__date__lte'}.get(key, key)
            query = query.filter(**{lookup: value})
        paginator = PageNumberPagination()
        paginator.page_size = 20
        page = paginator.paginate_queryset(query, request)
        return paginator.get_paginated_response([history_data(check) for check in page])


class CheckDetailView(APIView):
    def get_check(self, request, pk):
        return get_object_or_404(Check, pk=pk, user=request.user)

    def get(self, request, pk):
        return Response(history_data(self.get_check(request, pk)))

    def delete(self, request, pk):
        check = self.get_check(request, pk)
        check_id = check.pk
        check.delete()
        logger.info('analysis check_id=%s status=deleted', check_id)
        return Response(status=204)


class SummaryView(APIView):
    def get(self, request):
        query = Check.objects.all() if request.user.is_staff else Check.objects.filter(user=request.user)
        result = query.aggregate(sms_count=Count('id', filter=Q(type='sms')),
                                 url_count=Count('id', filter=Q(type='url')),
                                 average_risk_score=Avg('risk_score'))
        result['categories'] = dict(query.order_by().values_list('category').annotate(count=Count('id')))
        return Response(result)


class HealthView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        files = ('sms_model.joblib', 'sms_model_meta.json', 'url_model.joblib', 'url_model_meta.json')
        availability = {}
        for name in files:
            try:
                with (ARTIFACTS / name).open('rb') as stream:
                    availability[name] = 'available' if stream.read(1) else 'missing'
            except OSError:
                availability[name] = 'missing'
        return Response({'status': 'ok', 'models': availability, 'csrf_token': get_token(request)})
