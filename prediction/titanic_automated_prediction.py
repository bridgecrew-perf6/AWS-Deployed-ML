import pandas as pd
import numpy as np
import pickle


def predict_person(**kwargs):
    
    parameters_dict = pickle.load(open("prediction/outputs/parameters_dict.pkl","rb"))
    catboost_reg = pickle.load(open("prediction/outputs/catboost_ml.pkl", "rb"))

    if kwargs["cabin"] == '':
        kwargs['cabin'] = 'N'
    # Generate DataFrame from inputs
    df = pd.DataFrame.from_dict({"PassengerId":[kwargs["passengerid"]], "Pclass":[kwargs["pclass"]], "Name":[kwargs["name"]], "Sex":[kwargs["sex"]],
                                 "Age":[kwargs["age"]], "SibSp":[kwargs["sibsp"]], "Parch":[kwargs["parch"]], "Ticket":[kwargs["ticket"]], "Fare":[kwargs["fare"]], "Cabin":[kwargs["cabin"]],
                                 "Embarked":[kwargs["embarked"]]})

    
    # To map Sex feature
    gender_dictionary = {"female":1 , "male": 0}
    df.Sex = df.Sex.map(gender_dictionary)
    
    # Fill it if it is Null. Some of passengers had Null Cabin data, but same ticket number so they are matched.
    df["Cabin"].fillna('N', inplace=True)
    df["Cabin_Category"] = df["Cabin"].str[0]
    df["Cabin_Category"] = df.apply(lambda row: parameters_dict["ticket2cabin_dict"].get(row.Ticket, 'N'), axis=1)
    
    df["Cabin_Category"] = df["Cabin_Category"].apply(lambda x: 'A' if x=='T' else x)
    df["Cabin_Category"] = df["Cabin_Category"].apply(lambda x: 'BC' if (x=='B') or (x=='C')
                                    else ('DE' if x=='D' or x=='E'
                                         else('FG' if x=='F' or x=='G' else x)))
    
    # Since it is processed drop Cabin column
    df.drop(columns=["Cabin"], axis=1, inplace=True)
    
    # Obtain Title from name and group the passenger
    df["Title"] = df.Name.apply(lambda x: x[x.find(',') : x.find('.')][1:].strip())
    df["Title"] = df["Title"].apply(lambda x: "Miss" if x == 'Ms'or x == 'Mme' or x=='Mlle' else x)
    df["Title"] = df["Title"].apply(lambda x: "Other" if x != "Miss" and x != "Master" and x!= "Mr" and x!= "Mrs" else x)
    
    # Match passengers with same surname to clarify more Cabin data
    df["Surname"] = df.Name.apply(lambda x: x.lower().split(",")[0])
    df.Cabin_Category = df.apply(lambda x: parameters_dict["surname2cabin_dict"].get(x.Surname, "N") if x.Cabin_Category == "N" else x.Cabin_Category, axis=1)
    
    # Get total family population
    df["Ppl_in_Family"] = df["Parch"] + df["SibSp"] + 1
    df.drop(["Parch","SibSp"], axis=1, inplace=True)
    
    # Get travel group of the passenger by his/her ticket
    df["Ppl_in_Group"] = df.Ticket.apply(lambda x: parameters_dict["ppl_group_dict"].get(x, 1))
    
    # Encode person 
    ppl_dict = {1: "(0, 1]", 2:"(1, 2]", 3:"(2, 3]", 4:"(3, 4]", 5:"(5, 11]"}
    df["Ppl_in_Family"] = df["Ppl_in_Family"].map(ppl_dict)
    df["Ppl_in_Group"] = df["Ppl_in_Group"].map(ppl_dict)
    
    
    df["Group_Survive"] = df.PassengerId.apply(lambda x: parameters_dict["group_survive"].get(x, 0))
    
    def ticket_matcher(ticket):
        if ticket == "SOTONO2":
            ticket = "SOTONOQ"

        elif ticket == "A4":
            ticket = "A5"

        elif ticket == "STONO2" or ticket == "STONOQ":
            ticket = "STONO"

        return ticket
    
    # Remove punctional chars and encode ticket as Numeric if it starts with number.
    df["Ticket"] = df.Ticket.apply(lambda x: x.replace("/","").replace(".", "").split(" ")[0] if not x.isdigit() else "Numeric")
    df["Ticket"] = df.Ticket.apply(ticket_matcher)
    
    df["Ticket"] = df.Ticket.apply(lambda x: "Other" if x not in parameters_dict["ticket"] else x)
    
    # Suppress outlier Fare data by IQR.
    df.Fare = df.Fare.apply(lambda x: parameters_dict["fare_upper"] if x > parameters_dict["fare_upper"] else x)
    df.Fare = df.Fare.apply(lambda x: parameters_dict["fare_lower"] if x < parameters_dict["fare_lower"]  else x)
    
    # Suppress the outlier age values to interquartile ranges.  
    df.Age = df.Age.apply(lambda x: parameters_dict["age_upper"] if x > parameters_dict["age_upper"] else x)
    df.Age = df.Age.apply(lambda x: parameters_dict["age_lower"] if x < parameters_dict["age_lower"] else x)
    df.drop(["Name", "PassengerId","Surname"], axis=1, inplace=True)
    
    # Generate DataFrame with respect to train columns with the purpose of letting machine learning model to predict passenger properly.
    df_pred = pd.DataFrame(columns=parameters_dict["columns_list"])

    # Assign values to specific columns by each column.
    for i in range(len(df)):
        df_pred.loc[i] = 0
        # Ticket Assign
        for ea_ticket in parameters_dict["Ticket"].values:
            if df.at[i,"Ticket"] == ea_ticket[len("Ticket")+1:]:
                df_pred.at[i, ea_ticket] = 1
        
        # Title Assign
        for ea_title in parameters_dict["Title"].values:
            if df.at[i, "Title"] == ea_title[len("Title")+1:]:
                df_pred.at[i, ea_title] = 1
        
        #Ppl_in_Family Assign
        for ea_fam in parameters_dict["Ppl_in_Family"].values:
            if df.at[i, "Ppl_in_Family"] == ea_fam[len("Ppl_in_Family")+1:]:
                df_pred.at[i, ea_fam] = 1
        
        #Ppl_in_Group Assign
        for ea_gr in parameters_dict["Ppl_in_Group"].values:
            if df.at[i, "Ppl_in_Group"] == ea_gr[len("Ppl_in_Group")+1:]:
                df_pred.at[i, ea_gr] = 1
                
        #Embarked Assign
        for ea_em in parameters_dict["Embarked"].values:
            if df.at[i, "Embarked"] == ea_em[len("Embarked")+1:]:
                df_pred.at[i, ea_em] = 1
        
        # Cabin Assign
        for ea_cabin in parameters_dict["Cabin_Category"].values:
            if df.at[i, "Cabin_Category"] == ea_cabin[len("Cabin_Category")+1:]:
                df_pred.at[i, ea_cabin] = 1
        
        df_pred["Fare"] = df["Fare"]
        df_pred["Age"] = df["Age"]
        df_pred["Sex"] = df["Sex"]
    
    # Return True if the passenger is predicted as survived.
    if catboost_reg.predict(df_pred.to_numpy())[0] == 1:
        return True
    # Else return as not survived.
    return False
    


