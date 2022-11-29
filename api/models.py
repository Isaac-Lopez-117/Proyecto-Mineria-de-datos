from django.db import models

class Project(models.Model):
    title = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    desc = models.CharField(max_length=100, null=True)
    data = models.FileField(upload_to='data/')

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.data.delete()
        super().delete(*args, **kwargs)