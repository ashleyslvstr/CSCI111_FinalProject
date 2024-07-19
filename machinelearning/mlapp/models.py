from django.db import models
from django.utils import timezone

# Create your models here.
class Applicant(models.Model):

    GENDER_CHOICES = [
        ('0', 'Male'),
        ('1', 'Female'),
    ]

    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=6, choices=GENDER_CHOICES)
    car = models.BooleanField(default=False)
    property = models.BooleanField(default=False)
    workPhone = models.BooleanField(default=False)
    ownPhone = models.BooleanField(default=False)
    email = models.BooleanField(default=False)
    employment = models.BooleanField(default=False)
    children = models.PositiveIntegerField()
    family = models.PositiveIntegerField()
    duration = models.PositiveIntegerField()
    income = models.FloatField()
    employmentYears = models.PositiveIntegerField()
    created_at = models.DateTimeField(default=timezone.now)
    objects = models.Manager()

    def __str__(self):
        return f"Eligibility Form ID: {self.id}"