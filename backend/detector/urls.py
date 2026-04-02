from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('explain/', views.explain, name='explain'),
    path('health/', views.health_check, name='health'),
    path('debug-rss/', views.debug_rss, name='debug_rss'),
]