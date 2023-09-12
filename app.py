import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('users.csv')

"""-----------------------Plot Status Graph---------------------------"""
# # ======================================================================
# # >>|
# status_col = df.pop("status")  # To choose custom column to show
# sold_df = pd.DataFrame({"Status": status_col})  # Create a new DataFrame for "Sold" column
# #                         Y
# #                         |_______________________.      
# plt.figure()                                  # # |
# # bins means the size of grpah lines            # |
# #                                                 |
# #                                                 |
# #         ._______________________________________|
# #         |
# #         |
# #         v
# sold_df.Status.hist(bins=10)
# plt.show()
# # |<<

"""--------------------------Describe--------------------------------"""
# # =====================================================================
# # >>|

# print(df.describe())

# # |<<

"""------Count Number Of Offline Users  & Number Of Online Users ----"""
# # ====================================================================
# # >>|

# online_users = df[df["status"] == "Online"]
# offline_users = df[df["status"] == "Offline"]
# print("Number of online users:", len(online_users))
# print("Number of offline users:", len(offline_users))

# # |<<

"""----To arange users by  unique_id from largest to smallest---------"""
# # =====================================================================
# # >>|

# top_unique_id_users = df.nlargest(10, "unique_id")
# print("Top 5 users by Unique_ID:")
# print(top_unique_id_users)

# # |<<

"""--------To arange users by  unique_id from smallest to largest------"""
# # ======================================================================
# # >>|
# top_unique_id_users = df.nsmallest(10, "unique_id")
# print("Top 5 users by Unique_ID:")
# print(top_unique_id_users)
# # |<<

# ======||
# GRAPH ||
# ======||

"""--------To plot GRAPH of sold books by date"""
# # ====================================================================
# # >>|
# df_books = pd.read_csv('books.csv')
# df_books['Date'] = pd.to_datetime(df_books['Date'])
# plt.figure(figsize=(10, 6))
# plt.plot(df_books['Sold'], df_books['Date'], marker='o', linestyle='dashdot')
# plt.xlabel('Sold')
# plt.ylabel('Date')
# plt.title('Sold vs. Date')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # |<<

# ======||
# CHART ||
# ======||

"""--------To plot CHART of users who survived group by male and female """
# # ====================================================================
# # >>|

# df_titanic = pd.read_csv('train.csv')
# survived_col = df_titanic.pop('survived')
# plt.figure()
# pd.concat([df_titanic, survived_col], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()

# # |<<

CATEGORICAL_COLUMNS = ['role','status']
NUMERIC_COLUMNS = [ "id","unique_id","first_name","last_name","username","email","password","date"]
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS: # feature name is an index of categorical columns
    vocabulary = df[feature_name].unique() # to find unique category like offline or online
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    print(feature_columns)

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(df[feature_name].unique())  #to print the unique value in each column