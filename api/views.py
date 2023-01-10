from django.shortcuts import render, redirect
from django.views.generic import TemplateView

import os
from plotly.offline import plot
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np                     # Para crear vectores y matrices n dimensionales
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.svm import SVC                         #Support vector classifier
from kneed import KneeLocator

from sklearn.tree import DecisionTreeClassifier

from .forms import ProjectForm
from .models import Project

#Home de la pagina
class Home(TemplateView):
    template_name = 'home.html'

#Listado de todos los proyectos
def project_list(request):
    projects = Project.objects.all()
    print(request)
    return render(request, 'project_list.html', {
        'projects': projects
    })

#Creacion y subida de un proyecto nuevo
def upload_project(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('project_list')
    else:
        form = ProjectForm()
    
    return render(request, 'upload_project.html', { 'form': form})

#Eliminacion de un proyecto
def delete_project(request, pk):
    project = Project.objects.get(pk=pk)
    project.delete()
    return redirect('project_list')

#Metodo EDA
def eda(request):
    projects = Project.objects.all()
    return render(request, './EDA/EDA.html', {
        'projects': projects
    })

def eda_project(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        show_df = df[:10]
        size = df.shape

        #Tipo de data
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

        #Histogramas
        histograms = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            if df[column].dtypes != object:
                hist = px.histogram(df, x=df.columns[i])

                # Colocando valores de la grafica
                layout = {
                    'title': df.columns[i],
                    'xaxis_title': 'X',
                    'yaxis_title': 'Y',
                    'height': 420,
                    'width': 560,
                }
                    
                # Obteniento HTML para mostrar el histograma.
                plot_div = plot({'data': hist, 'layout': layout}, output_type='div')
                histograms.append({'data': plot_div})

        #Box
        boxes = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            if df[column].dtypes != object:
                box = px.box(df, x=df.columns[i])

                # Colocando valores de la grafica
                layout = {
                    'title': df.columns[i],
                    'xaxis_title': 'X',
                    'yaxis_title': 'Y',
                    'height': 420,
                    'width': 560,
                }
                    
                # Obteniento HTML para mostrar el histograma.
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
        #Validacion de variables categoricas
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
                    # Obteniento HTML para mostrar el histograma.
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
        # Colocando valores de la grafica
        layout_hm = {
            'title': project.name,
            'height': 420,
            'width': 560,
        }
        plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
        
        del(df)

    return render(request, './EDA/EDA_project.html', context={'histograms': histograms, 'boxes': boxes, 'project': project, 'df': show_df,
    'info': info, 'hm': plot_div_hm, 'dist': v_dist, 'agru': v_agru, 'vc': vc, 'size' : size, 'types': types, 'null': null})

#Metodo EDA
def pca(request):
    projects = Project.objects.all()
    return render(request, './PCA/PCA.html', {
        'projects': projects
    })

def pca_project(request, pk):
    if (request.method == 'POST') or (request.method == 'GET'):
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_") 
        show_df = df[:10]
        size = df.shape

        #Heatmap
        corr = df.corr()
        hm = px.imshow(corr, text_auto=True, aspect="auto")
        # Colocando valores de la grafica
        layout_hm = {
            'title': project.name,
            'height': 420,
            'width': 560,
        }
        plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')

        df_na = df.dropna()
        #Estandarizacion de datos 
        Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler
        valoresNum = df_na.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]) 
        MEstandarizada = Estandarizar.fit_transform(valoresNum)         # Se calculan la media y desviación para cada variable, y se escalan los datos
        std = pd.DataFrame(MEstandarizada, columns=valoresNum.columns)
        show_std = std[:10]

        pca = PCA(n_components=None)     #Se instancia el objeto PCA 
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
        # Colocando valores de la grafica
        layout_v = {
            'title': project.name,
            'height': 420,
            'width': 560,
        }
        plot_div_v = plot({'data': graph_var, 'layout': layout_v}, output_type='div')

        CargasComponentes = pd.DataFrame(abs(pca.components_), columns=valoresNum.columns)
        show_cc=CargasComponentes[:numComp]

        del(df)

    return render(request, './PCA/PCA_project.html', context={'project': project, 'df': show_df, 'size' : size, 'hm': plot_div_hm, 'col_num': valoresNum, 'std': show_std, 'comp': comp, 'var': varAcum, 
    'num_comp': numComp,'v': plot_div_v, 'cc': show_cc,})

def guardar(request, pk):
    project = Project.objects.get(pk=pk)
    source = project.data
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_") 
    c=request.POST.getlist('columnas')
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]
    #Se guarda el nuevo dataframe en un archivo CSV
    n_df.to_csv("api/media/data/newData.csv", index=False)
    del(df)
    return render(request, './PCA/PCA_final.html', context={'df': show_df})

#Metodo Arbol de decision
def ad(request):
    projects = Project.objects.all()
    return render(request, './ARBOLES/AD.html', {
        'projects': projects
    })

def ad_p0(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_")            
        show_df = df[:10]
        size = df.shape

        #Tipo de data
        types = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            value = df[column].dtypes
            types.append(str(column) + ':  ' + str(value))

        numericos = df.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"])
        
        n_df = numericos.dropna()

        #Creacion de un archivo CSV
        n_df.to_csv("api/media/data/paso_0.csv", index=False)

        del(df)
    return render(request, './ARBOLES/AD_p0.html', context={'project': project, 'df': show_df, 'size' : size, 'types': types, 'num': numericos})

def ad_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_0.csv"
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ", "_")    
    c=request.POST.getlist('columnas')
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]
    #Creacion de un archivo CSV
    n_df.to_csv("api/media/data/paso_1.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Colocando valores de la grafica
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
    del(df)
    return render(request, './ARBOLES/AD_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm})

def ad_p1_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_1.csv"
    df = pd.read_csv(source)
    
    entrada =request.POST.getlist('predictoras') #Obtencion de las variables predictoras
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]

    #Creacion de un archivo CSV
    df_X.to_csv("api/media/data/X.csv", index=False)
    
    salida=request.POST.getlist('salida') #Obtencion de la variable de salida
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    #Creacion de un archivo CSV
    df_Y.to_csv("api/media/data/Y.csv", index=False)
    del(df)

    if str(salida[0]) in entrada:
        return render(request, './ARBOLES/AD_p1_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})
    else:
        return render(request, 'ERROR.html') 
    

#Pronostico
def ad_p_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    #Division de valores
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_X, df_Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
    
    test_X = pd.DataFrame(X_test)
    show_test = test_X[:10]

    #Se entrena el modelo a partir de los datos de entrada
    PronosticoAD = DecisionTreeRegressor(random_state=0)
    entrenamiento = PronosticoAD.fit(X_train, Y_train)

    Y_Pronostico = entrenamiento.predict(X_test)

    ValoresMod = pd.DataFrame(Y_test, Y_Pronostico)

    score = r2_score(Y_test, Y_Pronostico) #Puntaje obtenido

    #Datos obtenidos del pronostico
    criterio = PronosticoAD.criterion
    importancia = pd.DataFrame({'Variable': list(df_X[list(df_X.columns)]),
                                'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
    estadisticas = []
    estadisticas.append("MAE: " + str(mean_absolute_error(Y_test, Y_Pronostico)))
    estadisticas.append("MSE: " + str(mean_squared_error(Y_test, Y_Pronostico)))
    estadisticas.append("RMSE: " + str(mean_squared_error(Y_test, Y_Pronostico, squared=False)))

    #Arbol generado
    reporte_arbol = export_text(PronosticoAD, feature_names = list(df_X.columns))
    rep_show = []
    rep_show = reporte_arbol.split("\n")

    return render(request, './ARBOLES/PRONOSTICO/AD_p_p2.html', context={'project': project, 'df': df_X, 'df_test': show_test, 'score': score, 'criterio': criterio, 'importancia': importancia, 
    "est": estadisticas, 'arbol': rep_show})

def ad_p_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_X, df_Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

    test_X = pd.DataFrame(X_test)
    show_test = test_X[:10]

    #Se entrena el modelo a partir de los datos de entrada
    PronosticoAD = DecisionTreeRegressor(random_state=0)
    entrenamiento = PronosticoAD.fit(X_train, Y_train)

    Y_Pronostico = entrenamiento.predict(X_test)

    ValoresMod = pd.DataFrame(Y_test, Y_Pronostico)

    #Prediccion en base a los datos de entrada dados por el usuario
    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [float(entradas[i])]})
    pronostico = pd.DataFrame(aux)
    resultado = PronosticoAD.predict(pronostico)

    score = r2_score(Y_test, Y_Pronostico)

    return render(request, './ARBOLES/PRONOSTICO/Prueba.html', context={'project': project, 'df': df_X, 'prueba': pronostico, 'resultado': resultado})

#Clasificacion
def ad_c_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    tam_1 = len(X_train)
    tam_2 = len(X_validation)

    Y_train = round(Y_train[Y_train.columns.values[0]])

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionAD = DecisionTreeClassifier(random_state=0)
    entrenamiendo = ClasificacionAD.fit(X_train, Y_train)

    Y_ClasificacionAD = entrenamiendo.predict(X_validation)
    Y_validation = round(Y_validation[Y_validation.columns.values[0]])

    ValoresMod = pd.DataFrame(Y_validation, Y_ClasificacionAD)

    score = accuracy_score(Y_validation, Y_ClasificacionAD)

    #Matriz de clasificación
    ModeloClasificacion = ClasificacionAD.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.values.ravel(), 
                                    ModeloClasificacion, 
                                    rownames=['Actual'], 
                                    colnames=['Clasificación']) 
    
    #Datos obtenidos de la clasificacion
    criterio = ClasificacionAD.criterion
    importancia = pd.DataFrame({'Variable': list(df_X[list(df_X.columns)]),
                                'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
    reporte = classification_report(Y_validation, Y_ClasificacionAD, output_dict=True)
    df_cr = pd.DataFrame(reporte).transpose()
    df_cr = df_cr.sort_values(by=['f1-score'], ascending=False)

    #Arbol generado
    reporte_arbol = export_text(ClasificacionAD, feature_names = list(df_X.columns))
    rep_show = []
    rep_show = reporte_arbol.split("\n")

    return render(request, './ARBOLES/CLASIFICACION/AD_c_p2.html', context={'project': project, 'df': df_X, 'X_train': tam_1, 'X_validation': tam_2, 'score': score, 'matriz': Matriz_Clasificacion,
    'criterio': criterio, 'importancia': importancia, 'reporte': df_cr, 'arbol': rep_show})

def ad_c_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    if len(entradas[0]) == 0:
        return render(request, 'temp.html')
    else:
        X = "api/media/data/X.csv"
        df_X = pd.read_csv(X)
        Y = "api/media/data/Y.csv"
        df_Y = pd.read_csv(Y)

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                    test_size = 0.2, 
                                                                                    random_state = 0,
                                                                                    shuffle = True)
        Y_train = round(Y_train[Y_train.columns.values[0]])

        #Se entrena el modelo a partir de los datos de entrada
        ClasificacionAD = DecisionTreeClassifier(random_state=0)
        entrenamiendo = ClasificacionAD.fit(X_train, Y_train)

        Y_ClasificacionAD = entrenamiendo.predict(X_validation)
        Y_validation = round(Y_validation[Y_validation.columns.values[0]])
        
        col = list(df_X.columns)
        aux = dict()
        for i in range(df_X.shape[1]):
            aux.update({col[i]: [float(entradas[i])]})
        clasificacion = pd.DataFrame(aux)
        resultado = ClasificacionAD.predict(clasificacion)

        temp = df_Y[df_Y.columns.values[0]].sort_values()
        salidas = temp.unique()

        #Area bajo la curva
        y_score = ClasificacionAD.predict_proba(X_validation)
        y_test_bin = label_binarize(Y_validation, classes=salidas)
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        AUC_list = []
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            if i == 0 :
                figAUC = px.line(x=fpr[i], y=tpr[i])
            else:
                figAUC.add_scatter(x=fpr[i], y=tpr[i])

            AUC_list.append('AUC para la clase {}: {}'.format(i+1, auc(fpr[i], tpr[i])))

        figAUC.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        plot_auc = plot({'data': figAUC}, output_type='div')
            
        return render(request, './ARBOLES/CLASIFICACION/Prueba.html', context={'project': project, 'df': df_X, 'prueba': clasificacion, 'resultado': resultado,
    'auc': plot_auc, 'auc_score':AUC_list})

#Metodo Bosque aleatorio
def ba(request):
    projects = Project.objects.all()
    return render(request, './BOSQUES/BA.html', {
        'projects': projects
    })

def ba_p0(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_") 
        show_df = df[:10]
        size = df.shape

        #Tipo de data
        types = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            value = df[column].dtypes
            types.append(str(column) + ':  ' + str(value))

        numericos = df.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"])
        
        n_df = numericos.dropna()

        #Creacion de un archivo CSV
        n_df.to_csv("api/media/data/paso_0.csv", index=False)

        del(df)
    return render(request, './BOSQUES/BA_p0.html', context={'project': project, 'df': show_df, 'size' : size, 'types': types,'num': numericos})

def ba_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_0.csv"
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ", "_")    
    c=request.POST.getlist('columnas')
    print(c)
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]

    n_df.to_csv("api/media/data/paso_1.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Colocando valores de la grafica
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
    del(df)
    return render(request, './BOSQUES/BA_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm})

def ba_p1_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_1.csv"
    df = pd.read_csv(source)

    entrada =request.POST.getlist('predictoras')
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]
    
    df_X.to_csv("api/media/data/X.csv", index=False)

    salida=request.POST.getlist('salida')
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    df_Y.to_csv("api/media/data/Y.csv", index=False)

    del(df)

    if str(salida[0]) in entrada:
        return render(request, './BOSQUES/BA_p1_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})
    else:
        return render(request, 'ERROR.html') 

#Pronostico
def ba_p_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_X, df_Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

    test_X = pd.DataFrame(X_test)
    show_test = test_X[:10]

    #Se entrena el modelo a partir de los datos de entrada
    PronosticoBA = RandomForestRegressor(random_state=0)
    entrenamiento = PronosticoBA.fit(X_train, Y_train)

    Y_Pronostico = entrenamiento.predict(X_test)

    ValoresMod = pd.DataFrame(Y_test, Y_Pronostico)

    score = r2_score(Y_test, Y_Pronostico)

    criterio = PronosticoBA.criterion
    importancia = pd.DataFrame({'Variable': list(df_X[list(df_X.columns)]),
                                'Importancia': PronosticoBA.feature_importances_}).sort_values('Importancia', ascending=False)
    estadisticas = []
    estadisticas.append("MAE: " + str(mean_absolute_error(Y_test, Y_Pronostico)))
    estadisticas.append("MSE: " + str(mean_squared_error(Y_test, Y_Pronostico)))
    estadisticas.append("RMSE: " + str(mean_squared_error(Y_test, Y_Pronostico, squared=False)))

    Estimador = PronosticoBA.estimators_[99]

    reporte_bosque = export_text(Estimador, feature_names = list(df_X.columns))
    rep_show = []
    rep_show = reporte_bosque.split("\n")

    return render(request, './BOSQUES/PRONOSTICO/BA_p_p2.html', context={'project': project, 'df': df_X, 'df_test': show_test, 'score': score, 'criterio': criterio, 'importancia': importancia, 
    "est": estadisticas, 'arbol': rep_show})

def ba_p_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_X, df_Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

    test_X = pd.DataFrame(X_test)
    show_test = test_X[:10]

    #Se entrena el modelo a partir de los datos de entrada
    PronosticoBA = RandomForestRegressor(random_state=0)
    entrenamiento = PronosticoBA.fit(X_train, Y_train)

    Y_Pronostico = entrenamiento.predict(X_test)

    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [float(entradas[i])]})
    pronostico = pd.DataFrame(aux)
    resultado = PronosticoBA.predict(pronostico)

    score = r2_score(Y_test, Y_Pronostico)

    return render(request, './BOSQUES/PRONOSTICO/Prueba.html', context={'project': project, 'df': df_X, 'prueba': pronostico, 'resultado': resultado})

#Clasificacion
def ba_c_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    tam_1 = len(X_train)
    tam_2 = len(X_validation)

    Y_train = round(Y_train[Y_train.columns.values[0]])

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionBA = RandomForestClassifier(random_state=0)
    entrenamiendo = ClasificacionBA.fit(X_train, Y_train)

    Y_ClasificacionBA = entrenamiendo.predict(X_validation)
    Y_validation = round(Y_validation[Y_validation.columns.values[0]])

    ValoresMod = pd.DataFrame(Y_validation, Y_ClasificacionBA)

    score = accuracy_score(Y_validation, Y_ClasificacionBA)

    #Matriz de clasificación
    ModeloClasificacion = ClasificacionBA.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.values.ravel(), 
                                    ModeloClasificacion, 
                                    rownames=['Reales'], 
                                    colnames=['Clasificación']) 
    
    criterio = ClasificacionBA.criterion
    importancia = pd.DataFrame({'Variable': list(df_X[list(df_X.columns)]),
                                'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
    reporte = classification_report(Y_validation, Y_ClasificacionBA, output_dict=True)
    df_cr = pd.DataFrame(reporte).transpose()
    df_cr = df_cr.sort_values(by=['f1-score'], ascending=False)

    Estimador = ClasificacionBA.estimators_[99]

    reporte_bosque = export_text(Estimador, feature_names = list(df_X.columns))
    rep_show = []
    rep_show = reporte_bosque.split("\n")

    return render(request, './BOSQUES/CLASIFICACION/BA_c_p2.html', context={'project': project, 'df': df_X, 'X_train': tam_1, 'X_validation': tam_2,'score': score, 'matriz': Matriz_Clasificacion,
    'criterio': criterio, 'importancia': importancia, 'reporte': df_cr, 'bosque': rep_show})

def ba_c_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

    Y_train = round(Y_train[Y_train.columns.values[0]])

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionBA = RandomForestClassifier(random_state=0)
    entrenamiendo = ClasificacionBA.fit(X_train, Y_train)

    Y_ClasificacionBA = entrenamiendo.predict(X_validation)
    Y_validation = round(Y_validation[Y_validation.columns.values[0]])
    
    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [float(entradas[i])]})
    clasificacion = pd.DataFrame(aux)
    resultado = ClasificacionBA.predict(clasificacion)
    
    temp = df_Y[df_Y.columns.values[0]].sort_values(ascending=False)
    salidas = temp.unique()
    
    #Area bajo la curva
    y_score = ClasificacionBA.predict_proba(X_validation)
    y_test_bin = label_binarize(Y_validation, classes=salidas)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    AUC_list = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        if i == 0 :
            figAUC = px.line(x=fpr[i], y=tpr[i])
        else:
            figAUC.add_scatter(x=fpr[i], y=tpr[i])
        
        AUC_list.append('AUC para la clase {}: {}'.format(i+1, auc(fpr[i], tpr[i])))

    figAUC.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    plot_auc = plot({'data': figAUC}, output_type='div')

    return render(request, './BOSQUES/CLASIFICACION/Prueba.html', context={'project': project, 'df': df_X, 'prueba': clasificacion, 'resultado': resultado, 
    'auc': plot_auc, 'auc_score':AUC_list})

#Segmentacion y clasificacion
def sc(request):
    projects = Project.objects.all()
    return render(request, './SC/SC.html', {
        'projects': projects
    })

def sc_p0(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_") 
        show_df = df[:10]
        size = df.shape

        #Tipo de data
        types = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            value = df[column].dtypes
            types.append(str(column) + ':  ' + str(value))

        numericos = df.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"])
        
        n_df = numericos.dropna()

        #Creacion de un archivo CSV
        n_df.to_csv("api/media/data/paso_0.csv", index=False)

        del(df)
    return render(request, './SC/SC_p0.html', context={'project': project, 'df': show_df, 'size' : size, 'types': types, 'num': n_df})

def sc_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_0.csv"
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_") 
    c=request.POST.getlist('columnas')
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]

    n_df.to_csv("api/media/data/paso_1.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Colocando valores de la grafica
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')

    df_na = df.dropna()
    #Estandarizacion de datos 
    Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler
    valoresNum = df_na.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]) 
    MEstandarizada = Estandarizar.fit_transform(valoresNum)         # Se calculan la media y desviación para cada variable, y se escalan los datos
    std = pd.DataFrame(MEstandarizada, columns=valoresNum.columns)
    show_std = std[:10]

    #Definición de k clusters para K-means
    #Se utiliza random_state para inicializar el generador interno de números aleatorios
    SSE = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)

    #Grafica del metodo del codo
    graph_var = px.line(SSE, markers=True)
    graph_var.update_xaxes(title_text='Cantidad de clusters *k*')
    graph_var.update_yaxes(title_text='SSE')
    
    # Colocando valores de la grafica
    layout_e = {
        'title': 'Elbow Method',
        'height': 420,
        'width': 560,
    }
    plot_elbow = plot({'data': graph_var, 'layout': layout_e}, output_type='div')

    kl = KneeLocator(range(2, 10), SSE, curve="convex", direction="decreasing")

    del(df)
    return render(request, './SC/SC_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm,'std': show_std, 'elbow': plot_elbow, 'clusters': kl.elbow})

def sc_p2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_1.csv"
    df = pd.read_csv(source)
    clusters=request.POST.getlist('clusters')
    show_df = df[:10]

    df_na = df.dropna()
    #Estandarizacion de datos 
    Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler
    valoresNum = df_na.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]) 
    MEstandarizada = Estandarizar.fit_transform(valoresNum)         # Se calculan la media y desviación para cada variable, y se escalan los datos
    
    #Se crean las etiquetas de los elementos en los clusters
    MParticional = KMeans(n_clusters=int(clusters[0]), random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    valores = MParticional.labels_
    
    df['clusterP'] = valores
    df.to_csv("api/media/data/clasificacion.csv", index=False)
    show_ndf = df[:10]

    #Cantidad de elementos en los clusters
    grupos = df.groupby(['clusterP'])['clusterP'].count()
    show_grupos = []
    for i in range(len(grupos)):
        show_grupos.append('Cluster [' + str(i) + ']: ' + str(grupos[i]))

    CentroidesP = df.groupby('clusterP').mean()

    # Gráfica de los elementos y los centros de los clusters
    figScatter = px.scatter_3d(data_frame = df, x = MEstandarizada[:,0], y = MEstandarizada[:,1], z = MEstandarizada[:,2], 
    color = df['clusterP'], hover_name = df['clusterP'], symbol = 'clusterP')
    figScatter.add_scatter3d(x = MParticional.cluster_centers_[:,0], y = MParticional.cluster_centers_[:,1], 
    z = MParticional.cluster_centers_[:,2], mode = 'markers')
    figScatter.update_layout(
        title = 'Gráfica de los elementos y los centros de los clusters'
    )
    plot_scatter = plot({'data': figScatter}, output_type = 'div')

    del(df)
    return render(request, './SC/SC_p2.html', context={'project': project, 'df': show_df, 'etiqueta': show_ndf, 'grupos': show_grupos, 'centroides': CentroidesP, 'plot': plot_scatter})

def sc_p2_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/clasificacion.csv"
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_") 

    entrada =request.POST.getlist('predictoras')
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]
    
    df_X.to_csv("api/media/data/X.csv", index=False)

    salida=request.POST.getlist('salida')
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    df_Y.to_csv("api/media/data/Y.csv", index=False)
    del(df)

    if str(salida[0]) in entrada:
        return render(request, './SC/SC_p2_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})
    else:
        return render(request, 'ERROR.html')

def sc_p3(request, pk):
    project = Project.objects.get(pk=pk)
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    tam_1 = len(X_train)
    tam_2 = len(X_validation)

    Y_train = round(Y_train[Y_train.columns.values[0]])

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionBA = RandomForestClassifier(random_state=0)
    entrenamiendo = ClasificacionBA.fit(X_train, Y_train)

    Y_ClasificacionBA = entrenamiendo.predict(X_validation)
    Y_validation = round(Y_validation[Y_validation.columns.values[0]])

    ValoresMod = pd.DataFrame(Y_validation, Y_ClasificacionBA)

    score = accuracy_score(Y_validation, Y_ClasificacionBA)

    #Matriz de clasificación
    ModeloClasificacion = ClasificacionBA.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.values.ravel(), 
                                    ModeloClasificacion, 
                                    rownames=['Reales'], 
                                    colnames=['Clasificación']) 
    
    criterio = ClasificacionBA.criterion
    importancia = pd.DataFrame({'Variable': list(df_X[list(df_X.columns)]),
                                'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
    reporte = classification_report(Y_validation, Y_ClasificacionBA, output_dict=True)
    df_cr = pd.DataFrame(reporte).transpose()
    df_cr = df_cr.sort_values(by=['f1-score'], ascending=False)

    Estimador = ClasificacionBA.estimators_[99]

    reporte_bosque = export_text(Estimador, feature_names = list(df_X.columns))
    rep_show = []
    rep_show = reporte_bosque.split("\n")

    return render(request, './SC/SC_p3.html', context={'project': project, 'df': df_X, 'X_train': tam_1, 'X_validation': tam_2,'score': score, 'matriz': Matriz_Clasificacion,
    'criterio': criterio, 'importancia': importancia, 'reporte': df_cr, 'bosque': rep_show})

def sc_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

    Y_train = round(Y_train[Y_train.columns.values[0]])

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionBA = RandomForestClassifier(random_state=0)
    entrenamiendo = ClasificacionBA.fit(X_train, Y_train)

    Y_ClasificacionBA = entrenamiendo.predict(X_validation)
    Y_validation = round(Y_validation[Y_validation.columns.values[0]])
    
    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [float(entradas[i])]})
    clasificacion = pd.DataFrame(aux)
    resultado = ClasificacionBA.predict(clasificacion)

    temp = df_Y[df_Y.columns.values[0]].sort_values()
    salidas = temp.unique()
    
    #Area bajo la curva
    y_score = ClasificacionBA.predict_proba(X_validation)
    y_test_bin = label_binarize(Y_validation, classes=salidas)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    AUC_list = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        if i == 0 :
            figAUC = px.line(x=fpr[i], y=tpr[i])
        else:
            figAUC.add_scatter(x=fpr[i], y=tpr[i])
        
        AUC_list.append('AUC para la clase {}: {}'.format(i+1, auc(fpr[i], tpr[i])))

    figAUC.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    plot_auc = plot({'data': figAUC}, output_type='div')


    return render(request, './SC/Prueba.html', context={'project': project, 'df': df_X, 'prueba': clasificacion, 'resultado': resultado, 'auc': plot_auc, 'auc_score': AUC_list})

#Maquinas de vectores de soporte
def svm(request):
    projects = Project.objects.all()
    return render(request, './SVM/SVM.html', {
        'projects': projects
    })

def svm_p0(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_")
        show_df = df[:10]
        size = df.shape

        #Tipo de data
        types = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            value = df[column].dtypes
            types.append(str(column) + ':  ' + str(value))

        numericos = df.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"])
        
        n_df = numericos.dropna()

        #Creacion de un archivo CSV
        n_df.to_csv("api/media/data/paso_0.csv", index=False)

        del(df)
    return render(request, './SVM/SVM_p0.html', context={'project': project, 'df': show_df, 'size' : size, 'types': types, 'num': n_df})

def svm_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_0.csv"
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ", "_")
    c=request.POST.getlist('columnas')
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]

    n_df.to_csv("api/media/data/paso_1.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Colocando valores de la grafica
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
    del(df)
    return render(request, './SVM/SVM_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm})

def svm_p1_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "api/media/data/paso_1.csv"
    df = pd.read_csv(source)

    entrada =request.POST.getlist('predictoras')
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]
    
    df_X.to_csv("api/media/data/X.csv", index=False)

    salida=request.POST.getlist('salida')
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    df_Y.to_csv("api/media/data/Y.csv", index=False)
    del(df)

    if str(salida[0]) in entrada:
        return render(request, './SVM/SVM_p1_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})
    else:
        return render(request, 'ERROR.html')

def svm_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)
    temp = request.POST.getlist('metodo')
    metodo = temp[0]
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    tam_1 = len(X_train)
    tam_2 = len(X_validation)
    
    Y_train = round(Y_train[Y_train.columns.values[0]])

    #Se declara el tipo de kernel y se entrena el modelo
    ModeloSVM = SVC(kernel=metodo)
    entrenamiento = ModeloSVM.fit(X_train, Y_train)

    Clasificaciones = entrenamiento.predict(X_validation)
    Y_validation = round(Y_validation[Y_validation.columns.values[0]])
    Clasificaciones = pd.DataFrame(Y_validation, Clasificaciones)

    #Se calcula la exactitud promedio de la validación
    exactitud = ModeloSVM.score(X_validation, Y_validation)

    #Validacion del modelo
    #Matriz de clasificación
    Clasificaciones = ModeloSVM.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.values.ravel(), 
                                    Clasificaciones, 
                                    rownames=['Real'], 
                                    colnames=['Clasificación'])  
    
    #Reporte de la clasificación
    reporte = classification_report(Y_validation, Clasificaciones, output_dict=True)
    df_cr = pd.DataFrame(reporte).transpose()
    df_cr = df_cr.sort_values(by=['f1-score'], ascending=False)
    
    VectoresSoporte = ModeloSVM.support_vectors_
    df_vs = pd.DataFrame(VectoresSoporte)
    show_df_vs = df_vs[:10]

    #Vectores de soporte
    n_vs = ModeloSVM.n_support_
    vs = ModeloSVM.support_

    return render(request, './SVM/SVM_p2.html', context={'project': project, 'metodo': metodo, 'df': df_X, 'X_train': tam_1, 'X_validation': tam_2, 'matriz': Matriz_Clasificacion,
    'score': exactitud, 'reporte': df_cr, 'df_vs': show_df_vs, 'n_vs': n_vs, 'vs': vs, })

def svm_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    X = "api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)
    temp = request.POST.getlist('metodo')
    metodo = temp[0]
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

    #Se declara el tipo de kernel y se entrena el modelo
    ModeloSVM = SVC(kernel=metodo)
    entrenamiento = ModeloSVM.fit(X_train, Y_train)

    Clasificaciones = entrenamiento.predict(X_validation)
    Clasificaciones = pd.DataFrame(Y_validation, Clasificaciones)

    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [float(entradas[i])]})
    clasificacion = pd.DataFrame(aux)
    resultado = ModeloSVM.predict(clasificacion)

    return render(request, './SVM/Prueba.html', context={'project': project, 'df': df_X, 'metodo': metodo, 'prueba': clasificacion, 'resultado': resultado,})

#Pagina "acerca de" de la aplicacion
def about(request):
    return render(request, 'about.html')