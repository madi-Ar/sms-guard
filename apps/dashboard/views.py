from django.shortcuts import render


def index(request):
    """Главная страница-заглушка, предусмотренная заданием 2."""
    return render(request, 'dashboard/index.html')
