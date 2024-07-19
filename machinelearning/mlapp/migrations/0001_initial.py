# Generated by Django 5.0.4 on 2024-07-19 14:40

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Applicant',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.PositiveIntegerField(max_length=3)),
                ('gender', models.CharField(choices=[('0', 'Male'), ('1', 'Female')], max_length=6)),
                ('car', models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('property', models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('workPhone', models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('ownPhone', models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('email', models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('employment', models.BooleanField(choices=[(True, 'Yes'), (False, 'No')])),
                ('children', models.PositiveIntegerField(max_length=2)),
                ('family', models.PositiveIntegerField(max_length=2)),
                ('duration', models.PositiveIntegerField(max_length=3)),
                ('income', models.FloatField()),
                ('employmentYears', models.PositiveIntegerField(max_length=2)),
                ('created_at', models.DateTimeField(blank=True, null=True)),
            ],
        ),
    ]
