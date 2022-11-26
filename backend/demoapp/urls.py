from django.contrib import admin
from django.urls import path, re_path
from api import views

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^api/project/$', views.project_list),
    re_path(r'^api/project/(?P<pk>\d+)$', views.project_detail),
]