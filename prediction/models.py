from email.policy import default
from django.db import models
from django import forms

class Data(models.Model):
    PassengerId = models.IntegerField(default=3)
    Pclass = models.IntegerField(default=3)
    Name = models.CharField(max_length=25, default="Heikkinen, Miss. Laina")
    Sex = models.CharField(max_length=25, default="female")
    Age = models.FloatField(default=26.0)
    SibSp = models.IntegerField(default=0)
    Parch = models.IntegerField(default=0)
    Ticket = models.CharField(max_length=25, default="STON/O2. 3101282")
    Fare = models.FloatField(default=24.00)
    Cabin = models.CharField(max_length=25,null=True, blank=True)
    Embarked = models.CharField(max_length=25, default="S")


    def __str__(self):
        return f"{self.title}"
