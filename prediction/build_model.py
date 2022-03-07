import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer
import pickle

import warnings
warnings.filterwarnings('ignore')

def main():
    
    # Parameters Dict
    parameters_dict = dict()
    
    # Load data sets
    df_train = pd.read_csv("datasets/train.csv")
    df_test = pd.read_csv("datasets/test.csv")
    
    # Combine train & test data to process by focusing all data.
    df = pd.concat([df_train, df_test], axis=0)
    df.reset_index(inplace=True, drop=True)
    
    # To map Sex feature
    gender_dictionary = {"female":1 , "male": 0}
    df.Sex = df.Sex.map(gender_dictionary)
    train_bound = len(df_train)
    
    
    df["Cabin"].fillna('N', inplace=True)
    df["Cabin_Category"] = df["Cabin"].str[0]
    
    # Generate cabin dictionary with respect to ticket numbers.
    ticket2cabin_dict = df[df.Cabin != 'N'][["Ticket", "Cabin_Category"]].drop_duplicates().set_index("Ticket").to_dict()["Cabin_Category"]
    df["Cabin_Category"] = df.apply(lambda row: ticket2cabin_dict.get(row.Ticket, 'N'), axis=1)
    parameters_dict["ticket2cabin_dict"] = ticket2cabin_dict

    # Fill it if it is Null. Some of passengers had Null Cabin data, but same ticket number so they are matched.
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
    
    surname2cabin_dict = df.groupby("Surname")["Cabin_Category"].apply(list).to_dict()
    for key in surname2cabin_dict:
        l = surname2cabin_dict[key]
        if "N" in l:
            l.remove("N")
        if len(l)>0:
            freq_char = max(set(l), key=l.count)
        else:
            freq_char = "N"
        surname2cabin_dict[key] = freq_char
    
    parameters_dict["surname2cabin_dict"] = surname2cabin_dict
    
    df.Cabin_Category = df.apply(lambda x: surname2cabin_dict.get(x.Surname, "N") if x.Cabin_Category == "N" else x.Cabin_Category, axis=1)
    
    # Get total family population
    df["Ppl_in_Family"] = df["Parch"] + df["SibSp"] + 1
    df.drop(["Parch","SibSp"], axis=1, inplace=True)
    
    ppl_group_dict = df["Ticket"].value_counts().to_dict()
    
    # Obtain travel group populations by matching tickets.
    df["Ppl_in_Group"] = df.Ticket.apply(lambda x: ppl_group_dict.get(x, 1))
    
    parameters_dict["ppl_group_dict"] = ppl_group_dict
    
    # Group passengers indexed their tickets. If they are alone they are encoded as 0.
    df['Ticket_Group_idx'] = df.Ticket.apply(lambda x: 0 if len(df.loc[df.Ticket == x, 'PassengerId'])==1
                                         else df.loc[df.Ticket == x, 'PassengerId'].min())

    
    group_fem_child = df.loc[((df.Sex == 1) | (df.Title == "Master")) & (df['Ticket_Group_idx'] > 0), 'Ticket_Group_idx'].value_counts()
    group_fem_child = group_fem_child[group_fem_child >= 2]


    # Grouping passengers if they have Master or female in their group
    df['Got_Fem_Child'] = 0
    df.loc[((df.Sex == 1) | (df.Title == "Master")) & (df['Ticket_Group_idx'] > 0)
           & (df["Ticket_Group_idx"].isin(group_fem_child.index)), 'Got_Fem_Child'] = 1

    df['Group_Survive'] = df.loc[df['Got_Fem_Child'] == 1].groupby('Ticket_Group_idx')['Survived'].transform(lambda x: x.mean(skipna=True))

    df.loc[df.Group_Survive.isnull(), 'Group_Survive'] = 0
    df['Group_Survive'] = df['Group_Survive'].astype(int)
    
    parameters_dict["group_survive"] = df.groupby("PassengerId").sum()["Group_Survive"].to_dict()

    df.drop(["Got_Fem_Child", "Ticket_Group_idx"], axis=1, inplace=True)
    
    # Remove punctional chars and encode all numeric tickets as Numeric
    df["Ticket"] = df.Ticket.apply(lambda x: x.replace("/","").replace(".", "").split(" ")[0] if not x.isdigit() else "Numeric")
    
    # Assume those tickets below are same.
    def ticket_matcher(ticket):
        if ticket == "SOTONO2":
            ticket = "SOTONOQ"

        elif ticket == "A4":
            ticket = "A5"

        elif ticket == "STONO2" or ticket == "STONOQ":
            ticket = "STONO"

        return ticket
    
    # After grouping tickets assign Other encoding to tickets which their total count below 15.
    df["Ticket"] = df.Ticket.apply(ticket_matcher)
    df["Ticket"] = df.Ticket.apply(lambda x: "Other" if len(df[df.Ticket == x]) < 15 else x)
    
    # Get all specific tickets.
    parameters_dict["ticket"] = df["Ticket"].unique()
    
    # Suppress outlier Fare data by IQR.
    Q1 = df.Fare.quantile(0.25)
    Q3 = df.Fare.quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5*IQR
    lower_bound = Q1 - 1.5*IQR
    
    parameters_dict["fare_upper"] = upper_bound
    parameters_dict["fare_lower"] = lower_bound
    
    df.Fare = df.Fare.apply(lambda x: upper_bound if x > upper_bound else x) 
    df.Fare = df.Fare.apply(lambda x: lower_bound if x < lower_bound else x)
    
    # Categorize population in group and family with respect to their counts.
    df["Ppl_in_Group"] = pd.cut(df["Ppl_in_Group"], [0, 1, 2, 3, 4, 5, df.Ppl_in_Group.max()])
    df["Ppl_in_Family"] = pd.cut(df["Ppl_in_Family"], [0, 1, 2, 3, 4,5, 11])
    
    # Save grouping ranges
    parameters_dict["ppl_in_group"] = [0, 1, 2, 3, 4, 5, df.Ppl_in_Group.max()]
    parameters_dict["ppl_in_family"] = [0, 1, 2, 3, 4, 5, df.Ppl_in_Family.max()]
    
    # One hot encoding
    df = pd.get_dummies(df, columns=["Title","Ppl_in_Family", "Ppl_in_Group", "Pclass","Ticket","Cabin_Category","Embarked"])
    df_spare = df[["PassengerId","Survived"]]
    df.drop(["Name","Survived", "PassengerId","Surname"], axis=1, inplace=True)
    
    # Get specific columns for one hot encodded features to use them at the prediction side
    parameters_dict["Title"] = df.columns[4:9]
    parameters_dict["Ppl_in_Family"] = df.columns[9:15]
    parameters_dict["Ppl_in_Group"] = df.columns[15:21]
    parameters_dict["Pclass"] = df.columns[21:24]
    parameters_dict["Ticket"] = df.columns[24:32]
    parameters_dict["Cabin_Category"] = df.columns[32:37]
    parameters_dict["Embarked"] = df.columns[37:40]
    
    parameters_dict["columns_list"] = df.columns
    
    # Fill missing age values with KKNImputer
    col_names = df.columns
    df2numpy = df.to_numpy()
    knn_imputer = KNNImputer(n_neighbors=3)
    df2numpy = knn_imputer.fit_transform(df2numpy)
    df = pd.DataFrame(df2numpy, columns=col_names)
    

    # Suppress the outlier age values to interquartile ranges.  
    Q1 = df.Age.quantile(0.25)
    Q3 = df.Age.quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5*IQR
    lower_bound = Q1 - 1.5*IQR
    
    parameters_dict["age_upper"] = upper_bound
    parameters_dict["age_lower"] = lower_bound
    
    df.Age = df.Age.apply(lambda x: upper_bound if x > upper_bound else x)
    df.Age = df.Age.apply(lambda x: lower_bound if x < lower_bound else x)
    
    # Insert back PassengerId and Survived data.
    df["Survived"] = df_spare.Survived
    df["PassengerId"] = df_spare.PassengerId
    
    # Cut labeled train data from the data set.
    df_train = df.iloc[:train_bound, :]
    #df_test = df.iloc[train_bound:, :]
    
    # Train the model
    x, y = df_train.drop(["Survived","PassengerId"], axis=1).to_numpy(), df_train["Survived"].to_numpy()
    catboost_clf = CatBoostClassifier(depth=9, iterations=1000, l2_leaf_reg=100, learning_rate=0.01, verbose=0,
                                  leaf_estimation_method="Newton").fit(x, y)
    
    # Save model
    with open("outputs/catboost_ml.pkl", "wb") as ml_f:
        pickle.dump(catboost_clf, ml_f)
    
    # Save parameters
    with open("outputs/parameters_dict.pkl", "wb") as param_f:
        pickle.dump(parameters_dict, param_f)
    
    print("Building section is completed!")
    
if __name__ == "__main__":
    main()