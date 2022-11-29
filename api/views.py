from django.shortcuts import render, redirect
from django.views.generic import TemplateView

from .forms import ProjectForm
from .models import Project

class Home(TemplateView):
    template_name = 'home.html'

def project_list(request):
    projects = Project.objects.all()
    return render(request, 'project_list.html', {
        'projects': projects
    })

def upload_project(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('project_list')
    else:
        form = ProjectForm()
    
    return render(request, 'upload_project.html', { 'form': form})

def delete_project(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        project.delete()
    return redirect('project_list')

def eda(request):
    projects = Project.objects.all()
    return render(request, 'EDA.html', {
        'projects': projects
    })

def eda_project(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
    return render(request, 'EDA_project.html', {
        'project': project
    })

def pca(request):
    projects = Project.objects.all()
    return render(request, 'PCA.html', {
        'projects': projects
    })

def pca_project(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
    return render(request, 'PCA_project.html', {
        'project': project
    })

def about(request):
    return render(request, 'about.html')