from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
#from .models import Applicant

# Create your views here.

# Machine Learning Functions
def knn_pred(X_train, x_test, Y_train):
    # Make a copy of the train and test features (to be free to modify such as to normalize)
    temp_X = X_train.copy()
    temp_X_test = x_test.copy()

    # Scale/Normalize this copy of the data
    for col in temp_X.columns:
        scaler = MinMaxScaler()
        temp_X[col] = scaler.fit_transform(temp_X[col].values.reshape(-1, 1))
        temp_X_test[col] = scaler.transform(temp_X_test[col].values.reshape(-1, 1))

    # Set the KNN as the model
    model = KNeighborsClassifier(n_neighbors=10)
    # Supply the model with the basis it can refer a pattern from
    model.fit(temp_X, Y_train)
    # Predict now the label of the test data (USER INPUT)
    y_pred = model.predict(temp_X_test)
    return "KNN: Applicant is eligible for a credit card." if y_pred == 1 else "KNN: Applicant is not eligible for a credit card."

def dectree_pred(X_train, x_test, Y_train):
    # Set the model
    tree = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0)
    # Supply the model with the basis it can refer a pattern from
    tree.fit(X_train, Y_train)
    # Predict now the label of the test data (USER INPUT)
    y_pred = tree.predict(x_test).round()
    return "Decision Tree: Applicant is eligible for a credit card." if y_pred == 1 else "Decision Tree: Applicant is not eligible for a credit card."


# Main function of the application
def index(request):
    if request.method == "POST":
        # Retrieve the data from index.html
        age = request.POST.get('fAge')
        gender = request.POST.get('fGender')
        car = request.POST.get('fCar') == '1'
        property = request.POST.get('fProperty') == '1'
        workPhone = request.POST.get('fWorkPhone') == '1'
        ownPhone = request.POST.get('fPersonalPhone') == '1'
        email = request.POST.get('fEmail') == '1'
        employment = request.POST.get('fEmployed') == '1'
        children = request.POST.get('fChildren')
        family = request.POST.get('fFamily')
        duration = request.POST.get('fDuration')
        income = request.POST.get('fIncome')
        employmentYears = request.POST.get('yrsEmployed')

        # Fix test data format for prediction
        data = [[gender, car, property, workPhone, ownPhone, email, employment, children, family, duration, income, age, employmentYears]]
        x_test = pd.DataFrame(data, columns=['Gender', 'Own_car', 'Own_property', 'Work_phone', 'Phone', 'Email', 'Unemployed', 'Num_children', 'Num_family', 'Account_length', 'Total_income', 'Age', 'Years_employed'])
        
        # Load training data
        train_csv_path = os.path.join(settings.STATICFILES_DIRS[0], 'train.csv')
        df = pd.read_csv(train_csv_path)
        X_train = df.drop(columns=['Target'])
        Y_train = df['Target']

        # Use the functions to make predictions
        knn_result = knn_pred(X_train, x_test, Y_train)
        dectree_result = dectree_pred(X_train, x_test, Y_train)
        return render(request, 'mlapp/index.html', {'knn_result': knn_result, 'dectree_result': dectree_result})

    return render(request, 'mlapp/index.html')