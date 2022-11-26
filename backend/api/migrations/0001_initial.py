# Generated by Django 4.1.3 on 2022-11-25 20:29

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=240, verbose_name='Name')),
                ('url', models.CharField(max_length=200, null=True)),
                ('desc', models.CharField(max_length=200, null=True)),
            ],
        ),
    ]
