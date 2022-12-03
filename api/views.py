from django.shortcuts import render, redirect
from django.views.generic import TemplateView

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        source = project.data
        df = pd.read_csv(source)
        show_df = df[:10]
        size = df.shape

        #Type of data
        types = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            value = df[column].dtypes
            types.append(str(column) + ':  ' + str(value))

        #Datos faltantes
        
        null = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            value = df[column].isnull().sum()
            null.append(str(column) + ':  ' + str(value))
        

        #Histograms
        histograms = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            if df[column].dtypes != object:
                hist = px.histogram(df, x=df.columns[i])

                # Setting layout of the figure.
                layout = {
                    'title': df.columns[i],
                    'xaxis_title': 'X',
                    'yaxis_title': 'Y',
                    'height': 420,
                    'width': 560,
                }
                    
                # Getting HTML needed to render the plot.
                plot_div = plot({'data': hist, 'layout': layout}, output_type='div')
                histograms.append({'data': plot_div})

        #Box
        boxes = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            if df[column].dtypes != object:
                hist = px.box(df, x=df.columns[i])

                # Setting layout of the figure.
                layout = {
                    'title': df.columns[i],
                    'xaxis_title': 'X',
                    'yaxis_title': 'Y',
                    'height': 420,
                    'width': 560,
                }
                    
                # Getting HTML needed to render the plot.
                plot_div = plot({'data': hist, 'layout': layout}, output_type='div')
                boxes.append({'data': plot_div})
    return render(request, 'EDA_project.html', context={'histograms': histograms, 'boxes': boxes, 'project': project, 'df': show_df, 'size' : size, 'types': types, 'null': null})

def pca(request):
    projects = Project.objects.all()
    return render(request, 'PCA.html', {
        'projects': projects
    })

def pca_project(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        df2 = df[:10]
        size = df.shape
    return render(request, 'PCA_project.html', context={'df': df2, 'size' : size})

def about(request):
    return render(request, 'about.html')