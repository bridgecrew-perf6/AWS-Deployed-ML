from django.forms import ModelForm, Textarea

from .models import Data

class DataForm(ModelForm):
    class Meta:
        model = Data
        fields = '__all__'
        widgets = {
            'body': Textarea()
        }