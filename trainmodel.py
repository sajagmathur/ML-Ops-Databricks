# Databricks notebook source
 !pip install snowflake

# COMMAND ----------

# %restart_python

# COMMAND ----------

# train.py
import os
import sys
# sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load and prepare data
import pandas as pd
import snowflake.connector

# Snowflake credentials (replace these with environment variables or secrets in production)
conn = snowflake.connector.connect(
    user='SAJAGMATHUR',
    password='Thati10pur@719',
    account='onmhvte-rm57820',  # your account locator (from URL)
    warehouse='COMPUTE_WH',
    database='ICECREAMDB',
    schema='PUBLIC'
)

# SQL query to load the table
sql_query = "SELECT * FROM ICECREAM"

# Execute the query and fetch into pandas
df = pd.read_sql(sql_query, conn)
X = df[['TEMP']]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Define output directory in DBFS
output_dir = "/Workspace/Users/sajag.mathur@exlservice.com/Model"
os.makedirs(output_dir, exist_ok=True)

# Save files to DBFS
joblib.dump(model, f"{output_dir}/model.pkl")
joblib.dump((X_test, y_test), f"{output_dir}/test_data.pkl")

print("✅ Model and test data saved to DBFS at /dbfs/tmp/model_output/")

# COMMAND ----------


