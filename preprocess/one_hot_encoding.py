import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import Utilities as Utils

df = pd.read_csv("C:\\Users\\bryan\\Desktop\\zhili\\ds\\the knack\\ds\\New folder\\breast.csv")

training_data,testing_data = Utils.train_test_split(df, test_size=0.2)

print("-----------Training Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(training_data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(training_data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(training_data))
print("-----------The null value summary-----------")
print(training_data.isnull().sum())

print("-----------Testing Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(testing_data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(testing_data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(testing_data))
print("-----------The null value summary-----------")
print(testing_data.isnull().sum())



X_train = df.loc[:,["Age recode with <1 year olds","Behavior code ICD-O-3","Breast - Adjusted AJCC 6th T (1988-2015)","Breast - Adjusted AJCC 6th N (1988-2015)","Breast - Adjusted AJCC 6th M (1988-2015)","CS tumor size (2004-2015)","CS extension (2004-2015)","CS lymph nodes (2004-2015)","CS mets at dx (2004-2015)","Histologic Type ICD-O-3","Laterality","Breast Subtype (2010+)","ER Status Recode Breast Cancer (1990+)","PR Status Recode Breast Cancer (1990+)","Derived HER2 Recode (2010+)","RX Summ--Surg Prim Site (1998+)","Radiation recode","Chemotherapy recode (yes, no/unk)","Marital status at diagnosis"]]
print("-----------The X_train row number-----------")
print(str(X_train.shape[0]))
print("-----------The X_train col number-----------")
print(str(X_train.shape[1]))

E_train = df.loc[:,"End Calc Vital Status (Adjusted)"]
print("-----------The E_train row number-----------")
print(str(E_train.shape[0]))
# print("-----------The E_train col number-----------")
# print(str(E_train.shape[1]))

Y_train = df.loc[:,"Number of Intervals (Calculated)"]
print("-----------The Y_train row number-----------")
print(str(Y_train.shape[0]))
# print("-----------The Y_train col number-----------")
# print(str(Y_train.shape[1]))

column_transformer = make_column_transformer((OneHotEncoder(),
                                       ["Age recode with <1 year olds","Behavior code ICD-O-3", "Breast - Adjusted AJCC 6th T (1988-2015)",
                                        "Breast - Adjusted AJCC 6th N (1988-2015)",
                                        "Breast - Adjusted AJCC 6th M (1988-2015)", "CS extension (2004-2015)",
                                        "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)",
                                        "Histologic Type ICD-O-3", "Laterality", "Breast Subtype (2010+)",
                                        "ER Status Recode Breast Cancer (1990+)",
                                        "PR Status Recode Breast Cancer (1990+)", "Derived HER2 Recode (2010+)",
                                        "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                        "Chemotherapy recode (yes, no/unk)", "Marital status at diagnosis"]),remainder="passthrough")
transform = column_transformer.fit_transform(X_train)
