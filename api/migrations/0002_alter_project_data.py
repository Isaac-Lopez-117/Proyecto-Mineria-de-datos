# Generated by Django 4.1.3 on 2023-01-04 18:18

import api.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='data',
            field=models.FileField(upload_to='data/', validators=[api.validators.validate_file_extension]),
        ),
    ]
