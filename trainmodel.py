# Databricks notebook source
 !pip install snowflake

# COMMAND ----------

# %restart_python

# COMMAND ----------

# train.py- updatd
import os
import sys
import dbutils
# sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load and prepare data
import pandas as pd
import snowflake.connector

# Snowflake credentials (replace these with environment variables or secrets in production)
# Retrieve notebook parameters passed via --notebook-params
user = dbutils.widgets.get("SNOWFLAKE_USER")
password = dbutils.widgets.get("SNOWFLAKE_PASSWORD")
account = dbutils.widgets.get("SNOWFLAKE_ACCOUNT")
warehouse = dbutils.widgets.get("SNOWFLAKE_WAREHOUSE")
database = dbutils.widgets.get("SNOWFLAKE_DATABASE")

# Optional: set them as environment variables if you need
os.environ["SNOWFLAKE_USER"] = user
os.environ["SNOWFLAKE_PASSWORD"] = password
os.environ["SNOWFLAKE_ACCOUNT"] = account
os.environ["SNOWFLAKE_WAREHOUSE"] = warehouse
os.environ["SNOWFLAKE_DATABASE"] = database

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=user,
    password=password,
    account=account,
    warehouse=warehouse,
    database=database,
    schema="PUBLIC"
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


