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

    path('proyectos/', views.project_list, name='project_list'),
    path('proyectos/upload/', views.upload_project, name='upload_project'),
    path('proyectos/<int:pk>/', views.delete_project, name='delete_project'),

    path('eda/', views.eda, name='eda'),
    path('eda/<int:pk>/', views.eda_project, name='eda_project'),

    path('pca/', views.pca, name='pca'),
    path('pca/<int:pk>/', views.pca_project, name='pca_project'),
    path('pca/final/<int:pk>', views.guardar, name='guardar'),

    path('ad/', views.ad, name='ad'),
    path('ad/<int:pk>/', views.ad_p0, name='ad_p0'),
    path('ad/1/<int:pk>/', views.ad_p1, name='ad_p1'),
    path('ad/1_2/<int:pk>/', views.ad_p1_2, name='ad_p1_2'),
    
    path('ad/pronostico/2/<int:pk>/', views.ad_p_p2, name='ad_p_p2'),
    path('ad/pronostico/final/<int:pk>/', views.ad_p_final, name='ad_p_final'),

    path('ad/clasificacion/2/<int:pk>/', views.ad_c_p2, name='ad_c_p2'),
    path('ad/clasificacion/final/<int:pk>/', views.ad_c_final, name='ad_c_final'),

    path('ba/', views.ba, name='ba'),
    path('ba/<int:pk>/', views.ba_p0, name='ba_p0'),
    path('ba/1/<int:pk>/', views.ba_p1, name='ba_p1'),
    path('ba/1_2/<int:pk>/', views.ba_p1_2, name='ba_p1_2'),
    
    path('ba/pronostico/2/<int:pk>/', views.ba_p_p2, name='ba_p_p2'),
    path('ba/pronostico/final/<int:pk>/', views.ba_p_final, name='ba_p_final'),

    path('ba/clasificacion/2/<int:pk>/', views.ba_c_p2, name='ba_c_p2'),
    path('ba/clasificacion/final/<int:pk>/', views.ba_c_final, name='ba_c_final'),

    path('sc/', views.sc, name='sc'),
    path('sc/<int:pk>/', views.sc_p0, name='sc_p0'),
    path('sc/1/<int:pk>/', views.sc_p1, name='sc_p1'),
    path('sc/2/<int:pk>/', views.sc_p2, name='sc_p2'),
    path('sc/2_2/<int:pk>/', views.sc_p2_2, name='sc_p2_2'),
    path('sc/3/<int:pk>/', views.sc_p3, name='sc_p3'),
    path('sc/final/<int:pk>/', views.sc_final, name='sc_final'),

    path('svm/', views.svm, name='svm'),
    path('svm/<int:pk>/', views.svm_p0, name='svm_p0'),
    path('svm/1/<int:pk>/', views.svm_p1, name='svm_p1'),
    path('svm/1_2/<int:pk>/', views.svm_p1_2, name='svm_p1_2'),
    path('svm/2/<int:pk>/', views.svm_p2, name='svm_p2'),
    path('svm/final/<int:pk>/', views.svm_final, name='svm_final'),

    path('about/', views.about, name='about'),
    path('admin/', admin.site.urls),
]
