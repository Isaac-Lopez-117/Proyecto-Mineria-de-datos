"""demoapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from api import views

urlpatterns = [
    path('', views.Home.as_view(), name='home'),

    path('projects/', views.project_list, name='project_list'),
    path('projects/upload/', views.upload_project, name='upload_project'),
    path('projects/<int:pk>/', views.delete_project, name='delete_project'),

    path('eda/', views.eda, name='eda'),
    path('eda/<int:pk>/', views.eda_project, name='eda_project'),

    path('pca/', views.pca, name='pca'),
    path('pca/<int:pk>/', views.pca_project, name='pca_project'),
    path('pca/final/<int:pk>', views.guardar, name='guardar'),

    path('ad/', views.ad, name='ad'),
    path('ad/pronostico/<int:pk>/', views.ad_pronostico, name='ad_p'),
    path('ad/pronostico/1/<int:pk>/', views.ad_p_p1, name='ad_p_p1'),
    path('ad/pronostico/1_2/<int:pk>/', views.ad_p_p1_2, name='ad_p_p1_2'),
    path('ad/pronostico/2/<int:pk>/', views.ad_p_p2, name='ad_p_p2'),
    path('ad/pronostico/final/<int:pk>/', views.ad_p_final, name='ad_p_final'),

    path('ad/clasificacion/<int:pk>/', views.ad_clasificacion, name='ad_c'),
    path('ad/clasificacion/1/<int:pk>/', views.ad_c_p1, name='ad_c_p1'),
    path('ad/clasificacion/1_2/<int:pk>/', views.ad_c_p1_2, name='ad_c_p1_2'),
    path('ad/clasificacion/2/<int:pk>/', views.ad_c_p2, name='ad_c_p2'),
    path('ad/clasificacion/final/<int:pk>/', views.ad_c_final, name='ad_c_final'),

    path('ba/', views.ba, name='ba'),
    path('ba/pronostico/<int:pk>/', views.ba_pronostico, name='ba_p'),
    path('ba/pronostico/1/<int:pk>/', views.ba_p_p1, name='ba_p_p1'),
    path('ba/pronostico/1_2/<int:pk>/', views.ba_p_p1_2, name='ba_p_p1_2'),
    path('ba/pronostico/2/<int:pk>/', views.ba_p_p2, name='ba_p_p2'),
    path('ba/pronostico/final/<int:pk>/', views.ba_p_final, name='ba_p_final'),

    path('ba/clasificacion/<int:pk>/', views.ba_clasificacion, name='ba_c'),
    path('ba/clasificacion/1/<int:pk>/', views.ba_c_p1, name='ba_c_p1'),
    path('ba/clasificacion/1_2/<int:pk>/', views.ba_c_p1_2, name='ba_c_p1_2'),
    path('ba/clasificacion/2/<int:pk>/', views.ba_c_p2, name='ba_c_p2'),
    path('ba/clasificacion/final/<int:pk>/', views.ba_c_final, name='ba_c_final'),

    path('about/', views.about, name='about'),
    path('admin/', admin.site.urls),
]
