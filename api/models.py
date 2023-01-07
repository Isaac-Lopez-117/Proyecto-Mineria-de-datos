from django.db import models
from .validators import validate_file_extension

class Project(models.Model):
    title = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    desc = models.CharField(max_length=100, null=True)
    data = models.FileField(upload_to='data/', validators=[validate_file_extension])

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.data.delete()
        super().delete(*args, **kwargs)