from django.urls import path
from django.contrib.auth import views as auth_views
from django.views.generic import TemplateView

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_view, name='predict'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
        path('data/wordcloud', views.wordcloud, name='wordcloud'),
        path('data/negation', views.negation, name='negation'),
        path('data/emotions', views.emotions, name='emotions'),
]
