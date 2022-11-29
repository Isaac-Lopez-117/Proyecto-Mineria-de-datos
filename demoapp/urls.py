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
    path('about/', views.about, name='about'),
    path('admin/', admin.site.urls),
]
