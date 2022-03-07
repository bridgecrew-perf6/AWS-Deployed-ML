from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from prediction.models import Data
from .forms import DataForm
from .titanic_automated_prediction import predict_person

# Create your views here.
def main(request):
    
    if request.method == "POST":
        form = DataForm(request.POST)
      
    else:
        form = DataForm()
    try:
        # Avoid any error by try and except. 
        # run predict_person function to get the information of passenger is survived or not.
        ml_pred = predict_person(passengerid=int(request.POST["PassengerId"]), pclass=int(request.POST["Pclass"]), name=request.POST["Name"], sex=request.POST["Sex"],
                         age=float(request.POST["Age"]), sibsp=int(request.POST["SibSp"]), parch=int(request.POST["Parch"]), ticket=request.POST["Ticket"],
                         fare=float(request.POST["Fare"]), cabin=request.POST["Cabin"], embarked=request.POST["Embarked"])
        # If survived return html of passenger is survived
        if  ml_pred:
            return render(request, "prediction/main_survived.html", {"form": form})

        # If not survived return html of passenger is not survived
        else:
            return render(request, "prediction/main_not_survived.html", {"form": form})

    except Exception as ex:
        # Generally at initialization of page it drops to error because of running post without an inputs. Therefore, basic html is returned.
        print(ex)
        return render(request, "prediction/main.html", {"form": form})