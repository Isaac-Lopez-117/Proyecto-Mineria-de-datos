from django.shortcuts import render, redirect
from django.views.generic import TemplateView

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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos


from sklearn.tree import DecisionTreeClassifier

from .forms import ProjectForm
from .models import Project

class Home(TemplateView):
    template_name = 'home.html'

def project_list(request):
    projects = Project.objects.all()
    print(request)
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

    return render(request, './EDA/EDA_project.html', context={'histograms': histograms, 'boxes': boxes, 'project': project, 'df': show_df,
    'info': info, 'hm': plot_div_hm, 'dist': v_dist, 'agru': v_agru, 'vc': vc, 'size' : size, 'types': types, 'null': null})

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

        df_na = df.dropna()
        #Estandarizacion de datos 
        Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler
        valoresNum = df_na.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]) 
        MEstandarizada = Estandarizar.fit_transform(valoresNum)         # Se calculan la media y desviación para cada variable, y se escalan los datos
        std = pd.DataFrame(MEstandarizada, columns=valoresNum.columns)
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

        del(df)

    return render(request, './PCA/PCA_project.html', context={'project': project, 'df': show_df, 'size' : size, 'hm': plot_div_hm, 'col_num': valoresNum, 'std': show_std, 'comp': comp, 'var': varAcum, 
    'num_comp': numComp,'v': plot_div_v, 'cc': show_cc,})

def guardar(request, pk):
    project = Project.objects.get(pk=pk)
    source = project.data
    df = pd.read_csv(source)
    c=request.POST.getlist('columnas')
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]
    n_df.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv", index=False)
    del(df)
    return render(request, './PCA/PCA_final.html', context={'df': show_df})

def ad(request):
    projects = Project.objects.all()
    return render(request, './ARBOLES/AD.html', {
        'projects': projects
    })

def ad_pronostico(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_")            
        show_df = df[:10]
        size = df.shape

        #Type of data
        types = []
        for i in range(df.shape[1]):
            column = df.columns.values[i]
            value = df[column].dtypes
            types.append(str(column) + ':  ' + str(value))

        del(df)
    return render(request, './ARBOLES/PRONOSTICO/AD_p.html', context={'project': project, 'df': show_df, 'size' : size, 'types': types,})

def ad_p_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = project.data
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ", "_")    
    c=request.POST.getlist('columnas')
    print(c)
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]

    n_df.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Setting layout of the figure.
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
    del(df)
    return render(request, './ARBOLES/PRONOSTICO/AD_p_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm})

def ad_p_p1_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv"
    df = pd.read_csv(source)

    entrada =request.POST.getlist('predictoras')
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]
    
    df_X.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv", index=False)

    salida=request.POST.getlist('salida')
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    df_Y.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv", index=False)
    del(df)
    return render(request, './ARBOLES/PRONOSTICO/AD_p_p1_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})

def ad_p_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
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

    score = r2_score(Y_test, Y_Pronostico)

    criterio = PronosticoAD.criterion
    importancia = pd.DataFrame({'Variable': list(df_X[list(df_X.columns)]),
                                'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
    estadisticas = []
    estadisticas.append("MAE: " + str(mean_absolute_error(Y_test, Y_Pronostico)))
    estadisticas.append("MSE: " + str(mean_squared_error(Y_test, Y_Pronostico)))
    estadisticas.append("RMSE: " + str(mean_squared_error(Y_test, Y_Pronostico, squared=False)))

    reporte_arbol = export_text(PronosticoAD, feature_names = list(df_X.columns))
    rep_show = []
    rep_show = reporte_arbol.split("\n")

    return render(request, './ARBOLES/PRONOSTICO/AD_p_p2.html', context={'project': project, 'df': df_X, 'df_test': show_test, 'score': score, 'criterio': criterio, 'importancia': importancia, 
    "est": estadisticas, 'arbol': rep_show})

def ad_p_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
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

    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [int(entradas[i])]})
    pronostico = pd.DataFrame(aux)
    resultado = PronosticoAD.predict(pronostico)

    score = r2_score(Y_test, Y_Pronostico)

    return render(request, './ARBOLES/PRONOSTICO/Prueba.html', context={'project': project, 'df': df_X, 'prueba': pronostico, 'resultado': resultado})

def ad_clasificacion(request, pk):
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

        del(df)
    return render(request, './ARBOLES/CLASIFICACION/AD_c.html', context={'project': project, 'df': show_df, 'size' : size, 'types': types, })

def ad_c_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = project.data
    df = pd.read_csv(source)
    c=request.POST.getlist('columnas')
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]

    n_df.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Setting layout of the figure.
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
    del(df)
    return render(request, './ARBOLES/CLASIFICACION/AD_c_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm})

def ad_c_p1_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv"
    df = pd.read_csv(source)

    entrada =request.POST.getlist('predictoras')
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]
    
    df_X.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv", index=False)

    salida=request.POST.getlist('salida')
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    df_Y.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv", index=False)
    del(df)
    return render(request, './ARBOLES/CLASIFICACION/AD_c_p1_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})

def ad_c_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    tam_1 = len(X_train)
    tam_2 = len(X_validation)

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionAD = DecisionTreeClassifier(random_state=0)
    entrenamiendo = ClasificacionAD.fit(X_train, Y_train)

    Y_ClasificacionAD = entrenamiendo.predict(X_validation)

    ValoresMod = pd.DataFrame(Y_validation, Y_ClasificacionAD)

    score = accuracy_score(Y_validation, Y_ClasificacionAD)

    #Matriz de clasificación
    ModeloClasificacion = ClasificacionAD.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.values.ravel(), 
                                    ModeloClasificacion, 
                                    rownames=['Actual'], 
                                    colnames=['Clasificación']) 
    
    criterio = ClasificacionAD.criterion
    importancia = pd.DataFrame({'Variable': list(df_X[list(df_X.columns)]),
                                'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
    reporte = classification_report(Y_validation, Y_ClasificacionAD, output_dict=True)
    df_cr = pd.DataFrame(reporte).transpose()
    df_cr = df_cr.sort_values(by=['f1-score'], ascending=False)

    reporte_arbol = export_text(ClasificacionAD, feature_names = list(df_X.columns))
    rep_show = []
    rep_show = reporte_arbol.split("\n")

    return render(request, './ARBOLES/CLASIFICACION/AD_c_p2.html', context={'project': project, 'df': df_X, 'X_train': tam_1, 'X_validation': tam_2,'score': score, 'matriz': Matriz_Clasificacion,
    'criterio': criterio, 'importancia': importancia, 'reporte': df_cr, 'arbol': rep_show})

def ad_c_final(request, pk):
    project = Project.objects.get(pk=pk)
    entradas =request.POST.getlist('entradas')
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionAD = DecisionTreeClassifier(random_state=0)
    entrenamiendo = ClasificacionAD.fit(X_train, Y_train)

    Y_ClasificacionAD = entrenamiendo.predict(X_validation)
    
    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [int(entradas[i])]})
    clasificacion = pd.DataFrame(aux)
    resultado = ClasificacionAD.predict(clasificacion)
    score = accuracy_score(Y_validation, Y_ClasificacionAD)

    return render(request, './ARBOLES/CLASIFICACION/Prueba.html', context={'project': project, 'df': df_X, 'prueba': clasificacion, 'resultado': resultado})

def ba(request):
    projects = Project.objects.all()
    return render(request, './BOSQUES/BA.html', {
        'projects': projects
    })

def ba_pronostico(request, pk):
    if request.method == 'POST':
        project = Project.objects.get(pk=pk)
        source = project.data
        df = pd.read_csv(source)
        for i in range(df.shape[1]):
            df.columns.values[i] = df.columns.values[i].replace(" ", "_") 
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

        del(df)
    return render(request, './BOSQUES/PRONOSTICO/BA_p.html', context={'project': project, 'df': show_df, 'size' : size, })

def ba_p_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = project.data
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ", "_")    
    c=request.POST.getlist('columnas')
    print(c)
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]

    n_df.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Setting layout of the figure.
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
    del(df)
    return render(request, './BOSQUES/PRONOSTICO/BA_p_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm})

def ba_p_p1_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv"
    df = pd.read_csv(source)

    entrada =request.POST.getlist('predictoras')
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]
    
    df_X.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv", index=False)

    salida=request.POST.getlist('salida')
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    df_Y.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv", index=False)
    del(df)
    return render(request, './BOSQUES/PRONOSTICO/BA_p_p1_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})

def ba_p_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
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
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
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
        aux.update({col[i]: [int(entradas[i])]})
    pronostico = pd.DataFrame(aux)
    resultado = PronosticoBA.predict(pronostico)

    score = r2_score(Y_test, Y_Pronostico)

    return render(request, './BOSQUES/PRONOSTICO/Prueba.html', context={'project': project, 'df': df_X, 'prueba': pronostico, 'resultado': resultado})

def ba_clasificacion(request, pk):
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

        del(df)
    return render(request, './BOSQUES/CLASIFICACION/BA_c.html', context={'project': project, 'df': show_df, 'size' : size, 'types': types, })

def ba_c_p1(request, pk):
    project = Project.objects.get(pk=pk)
    source = project.data
    df = pd.read_csv(source)
    c=request.POST.getlist('columnas')
    n_df = df.drop(df[c], axis=1)
    show_df = n_df[:10]

    n_df.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv", index=False)

    #Heatmap
    corr = n_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto")
    # Setting layout of the figure.
    layout_hm = {
        'title': project.name,
        'height': 420,
        'width': 560,
    }
    plot_div_hm = plot({'data': hm, 'layout': layout_hm}, output_type='div')
    del(df)
    return render(request, './BOSQUES/CLASIFICACION/BA_c_p1.html', context={'project': project, 'df': show_df, 'hm': plot_div_hm})

def ba_c_p1_2(request, pk):
    project = Project.objects.get(pk=pk)
    source = "C:/Users/USER/Desktop/Proyecto/api/media/data/newData.csv"
    df = pd.read_csv(source)

    entrada =request.POST.getlist('predictoras')
    n_df = df.drop(df[entrada], axis=1)
    X = np.array(n_df[list(n_df.columns)])
    df_X = pd.DataFrame(data=X, columns=n_df.columns.values)
    show_df_x = df_X[:10]
    
    df_X.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv", index=False)

    salida=request.POST.getlist('salida')
    Y = np.array(df[salida])
    df_Y = pd.DataFrame(data=Y, columns=salida)
    show_df_y = df_Y[:10]
    
    df_Y.to_csv("C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv", index=False)
    del(df)
    return render(request, './BOSQUES/CLASIFICACION/BA_c_p1_2.html', context={'project': project, 'X': show_df_x, 'Y': show_df_y,})

def ba_c_p2(request, pk):
    project = Project.objects.get(pk=pk)
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    tam_1 = len(X_train)
    tam_2 = len(X_validation)

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionBA = RandomForestClassifier(random_state=0)
    entrenamiendo = ClasificacionBA.fit(X_train, Y_train)

    Y_ClasificacionBA = entrenamiendo.predict(X_validation)

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
    X = "C:/Users/USER/Desktop/Proyecto/api/media/data/X.csv"
    df_X = pd.read_csv(X)
    Y = "C:/Users/USER/Desktop/Proyecto/api/media/data/Y.csv"
    df_Y = pd.read_csv(Y)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(df_X, df_Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

    #Se entrena el modelo a partir de los datos de entrada
    ClasificacionBA = RandomForestClassifier(random_state=0)
    entrenamiendo = ClasificacionBA.fit(X_train, Y_train)

    Y_ClasificacionBA = entrenamiendo.predict(X_validation)
    
    col = list(df_X.columns)
    aux = dict()
    for i in range(df_X.shape[1]):
        aux.update({col[i]: [int(entradas[i])]})
    clasificacion = pd.DataFrame(aux)
    resultado = ClasificacionBA.predict(clasificacion)
    score = accuracy_score(Y_validation, Y_ClasificacionBA)

    return render(request, './ARBOLES/CLASIFICACION/Prueba.html', context={'project': project, 'df': df_X, 'prueba': clasificacion, 'resultado': resultado})

def about(request):
    projects = Project.objects.all()
    return render(request, 'about.html', {
        'projects': projects
    })