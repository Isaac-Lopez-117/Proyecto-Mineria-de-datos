from django.shortcuts import render, redirect
from django.views.generic import TemplateView

from plotly.offline import plot
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np                     # Para crear vectores y matrices n dimensionales

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

        #Estadistica
        info = df.describe()

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
                box = px.box(df, x=df.columns[i])

                # Setting layout of the figure.
                layout = {
                    'title': df.columns[i],
                    'xaxis_title': 'X',
                    'yaxis_title': 'Y',
                    'height': 420,
                    'width': 560,
                }
                    
                # Getting HTML needed to render the plot.
                plot_div = plot({'data': box, 'layout': layout}, output_type='div')
                boxes.append({'data': plot_div})
        
        #Variables categoricas
        #Distribucion
        confirm = False
        v_dist = []
        vc = null
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            if df[column].dtypes == object:
                confirm = True
        
        if confirm == True:
            vc = df.describe(include='object')
            for col in df.select_dtypes(include='object'):
                if df[col].nunique()<10:
                    dist = px.histogram(df, y=col)
                    layout = {
                        'title': col,
                        'height': 420,
                        'width': 560,
                    }
                    # Getting HTML needed to render the plot.
                    plot_div = plot({'data': dist, 'layout': layout}, output_type='div')
                    v_dist.append({'data': plot_div})

        #Agrupacion
        v_agru = []
        for col in df.select_dtypes(include='object'):
            if df[col].nunique() < 10:
                agru = df.groupby(col).agg(['mean'])
                v_agru.append(agru)
        
        #Heatmap
        corr = df.corr()
        hm = px.imshow(corr, text_auto=True, aspect="auto")
        # Setting layout of the figure.
        layout_hm = {
            'title': project.name,
            'height': 420,
            'width': 560,
        }
        plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
        
        del(df)

    return render(request, 'EDA_project.html', context={'histograms': histograms, 'boxes': boxes, 'project': project, 'df': show_df,
    'info': info, 'hm': plot_div_hm, 'dist': v_dist, 'agru': v_agru, 'vc': vc, 'size' : size, 'types': types, 'null': null})

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
        show_df = df[:10]
        size = df.shape

        #Heatmap
        corr = df.corr()
        hm = px.imshow(corr, text_auto=True, aspect="auto")
        # Setting layout of the figure.
        layout_hm = {
            'title': project.name,
            'height': 420,
            'width': 560,
        }
        plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')


        #Estandarizacion de datos 
        Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler
        valoresNum = df.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]) 
        MEstandarizada = Estandarizar.fit_transform(df)         # Se calculan la media y desviación para cada variable, y se escalan los datos
        std = pd.DataFrame(MEstandarizada, columns=df.columns)
        show_std = std[:10]

        pca = PCA(n_components=10)     #Se instancia el objeto PCA 
        pca.fit(MEstandarizada)        #Se obtiene los componentes
        comp = pca.components_

        #Seleccion de componentes
        var = pca.explained_variance_ratio_
        numComp = 0
        while sum(var[0:numComp]) < 0.9:
            varAcum = sum(var[0:numComp])
            numComp = numComp + 1
        #Se le resta uno al numero de componentes, debido a que el ciclo while fuerza a sumar un componente de mas al final
        numComp = numComp - 1

        #Grafica de varianza
        graph_var = px.line(y=np.cumsum(pca.explained_variance_ratio_))
        graph_var.update_xaxes(title_text='Numero de componentes')
        graph_var.update_yaxes(title_text='Varianza acumulada')
        # Setting layout of the figure.
        layout_v = {
            'title': project.name,
            'height': 420,
            'width': 560,
        }
        plot_div_v = plot({'data': graph_var, 'layout': layout_v}, output_type='div')

        CargasComponentes = pd.DataFrame(abs(pca.components_), columns=valoresNum.columns)
        show_cc=CargasComponentes[:numComp]

        muestra = 0.50
        n_df = pd.DataFrame()
        for i in range(show_cc.shape[1]):
            column = show_cc.columns.values[i]
            if np.any(show_cc[column].values > muestra) == False:
                n_df = df.drop(columns=[column])
        show_ndf = n_df[:10]
        del(df)
        n_df.to_csv("C:/Users/USER/Desktop/test/api/media/data/newData.csv")
        

    return render(request, 'PCA_project.html', context={'project': project, 'df': show_df, 'size' : size, 'hm': plot_div_hm, 'std': show_std, 'comp': comp, 'var': varAcum, 
    'num_comp': numComp,'v': plot_div_v, 'cc': show_cc, 'ndf': show_ndf, })

def about(request):
    return render(request, 'about.html')