from django.db import models

# Create your models here.
class Applicant(models.Model):

    GENDER_CHOICES = [
        ('0', 'Male'),
        ('1', 'Female'),
    ]

    YES_NO_CHOICES = [
        (True, 'Yes'),
        (False, 'No')
    ]

    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=6, choices=GENDER_CHOICES)
    car = models.BooleanField(choices=YES_NO_CHOICES)
    property = models.BooleanField(choices=YES_NO_CHOICES)
    workPhone = models.BooleanField(choices=YES_NO_CHOICES)
    ownPhone = models.BooleanField(choices=YES_NO_CHOICES)
    email = models.BooleanField(choices=YES_NO_CHOICES)
    employment = models.BooleanField(choices=YES_NO_CHOICES)
    children = models.PositiveIntegerField()
    family = models.PositiveIntegerField()
    duration = models.PositiveIntegerField()
    income = models.FloatField()
    employmentYears = models.PositiveIntegerField()
    created_at = models.DateTimeField(blank=True, null=True)
    objects = models.Manager()

    def __str__(self):
        return f"Eligibility Form ID: {self.id}"