import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
from django.db import models

class Project(models.Model):
    name = models.CharField("Name", max_length=240)
    url = models.CharField(max_length=200, null=True)
    desc = models.CharField(max_length=200, null=True)

    data = None
    rows = None
    cols = None
    code = None
    dataGraph = None
    resType = None
    
    def loadData(self):
        self.data = pd.read_csv(self.url)

    def __str__(self):
        return self.name
