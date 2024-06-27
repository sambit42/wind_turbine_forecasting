# Databricks notebook source
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
from sparkmeasure import TaskMetrics
taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

try : 
    env = dbutils.widgets.get("env")
    task = dbutils.widgets.get("task")
except :
    env, task = "dev","fe"
print(f"Input environment : {env}")
print(f"Input task : {task}")

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
import ast
from MLCORE_SDK import mlclient
from pyspark.sql import functions as F
import pickle

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = ast.literal_eval(solution_config)
except Exception as e:
    print(e)
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

# JOB SPECIFIC PARAMETERS FOR FEATURE PIPELINE
if task.lower() == "fe":
    batch_size = int(solution_config["feature_pipelines_ft"].get("batch_size",500))
    input_table_configs = solution_config["feature_pipelines_ft"]["datalake_configs"]["input_tables"]
    output_table_configs = solution_config["feature_pipelines_ft"]["datalake_configs"]['output_tables']
    is_scheduled = solution_config["feature_pipelines_ft"]["is_scheduled"]
    cron_job_schedule = solution_config["feature_pipelines_ft"].get("cron_job_schedule","0 */10 * ? * *")
else:
    # JOB SPECIFIC PARAMETERS FOR DATA PREP DEPLOYMENT
    batch_size = int(solution_config["data_prep_deployment_ft"].get("batch_size",500))
    input_table_configs = solution_config["data_prep_deployment_ft"]["datalake_configs"]["input_tables"]
    output_table_configs = solution_config["data_prep_deployment_ft"]["datalake_configs"]['output_tables']
    is_scheduled = solution_config["data_prep_deployment_ft"]["is_scheduled"]
    cron_job_schedule = solution_config["data_prep_deployment_ft"].get("cron_job_schedule","0 */10 * ? * *")

# COMMAND ----------

def get_name_space(table_config):
    data_objects = {}
    for table_name, config in table_config.items() : 
        catalog_name = config.get("catalog_name", None)
        schema = config.get("schema", None)
        table = config.get("table", None)

        if catalog_name and catalog_name.lower() != "none": 
            table_path = f"{catalog_name}.{schema}.{table}"
        else :
            table_path = f"{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

source_1_df = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")

# COMMAND ----------

if is_scheduled:
  pickle_file_path = f"/mnt/FileStore/{output_table_configs['output_1']['schema']}"
  dbutils.fs.mkdirs(pickle_file_path)
  print(f"Created directory : {pickle_file_path}")
  pickle_file_path = f"/dbfs/{pickle_file_path}/{output_table_configs['output_1']['table']}.pickle"

  try : 
    with open(pickle_file_path, "rb") as handle:
        obj_properties = pickle.load(handle)
        print(f"Instance loaded successfully")
  except Exception as e:
    print(f"Exception while loading cache : {e}")
    obj_properties = {}
  print(f"Existing Cache : {obj_properties}")

  if not obj_properties :
    start_marker = 1
  elif obj_properties and obj_properties.get("end_marker",0) == 0:
    start_marker = 1
  else :
    start_marker = obj_properties["end_marker"] + 1
  end_marker = start_marker + batch_size - 1
else :
  start_marker = 1
  end_marker = source_1_df.count()

print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

source_1_df = source_1_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

if not source_1_df.first():
  dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

if task.lower() != "fe":
    # Calling job run add for DPD job runs
    mlclient.log(
        operation_type="job_run_add", 
        session_id = sdk_session_id, 
        dbutils = dbutils, 
        request_type = task, 
        job_config = 
        {
            "table_name" : output_table_configs["output_1"]["table"],
            "table_type" : "Source",
            "batch_size" : batch_size
        },
        tracking_env = env,
        spark = spark,
        verbose = True,
        )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### FEATURE ENGINEERING

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### FEATURE ENGINEERING on Feature Data

# COMMAND ----------

data = source_1_df.toPandas()

# COMMAND ----------

import pandas as pd
data = data.drop(columns=['date_time'])

# COMMAND ----------

end_date = pd.to_datetime('2022-01-01 00:00:00')  # Set your desired end date
freq = '10T'  # '10T' represents a 10-minute frequency

# Calculate the start date
start_date = end_date - pd.to_timedelta((len(data) - 1) * pd.to_timedelta(freq))

# Step 2: Create a new date column with 10-minute frequency
date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

# Step 3: Add the new date column to the DataFrame
data['date_time'] = date_range[:len(data)]

# COMMAND ----------

data.display()

# COMMAND ----------

data["date_time"].value_counts()

# COMMAND ----------

data.info()

# COMMAND ----------

data.isnull().sum()

# COMMAND ----------

df_wind = data.drop(['ControlBoxTemperature'], axis=1)

# COMMAND ----------

df_wind.display()

# COMMAND ----------

df_wind = df_wind.sort_values(by='date_time')

# COMMAND ----------

df_wind.display()

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
# Define the columns to exclude from min-max scaling
exclude_columns = ['index', 'date_time', 'timestamp', 'date', 'id']

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Apply min-max scaling to the DataFrame while excluding specified columns
scaled_columns = df_wind.drop(columns=exclude_columns).columns
df_wind[scaled_columns] = scaler.fit_transform(df_wind[scaled_columns])

# COMMAND ----------

df_wind['day'] = df_wind['date_time'].dt.day
df_wind['month'] = df_wind['date_time'].dt.month
df_wind["day_of_week"] = df_wind['date_time'].dt.strftime('%A')
df_wind['quarter'] = df_wind['date_time'].dt.quarter
quarter_mapping = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
df_wind['quarter'] = df_wind['quarter'].map(quarter_mapping)

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_wind['quarter'] = le.fit_transform(df_wind['quarter'])
df_wind['day_of_week'] = le.fit_transform(df_wind['day_of_week'])

# COMMAND ----------

df_wind.display()

# COMMAND ----------

output_1_df = spark.createDataFrame(df_wind)

# COMMAND ----------

output_1_df = output_1_df.drop('date','timestamp')

# COMMAND ----------

output_1_df.display()

# COMMAND ----------

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats = (
        "MM-dd-yyyy",
        "dd-MM-yyyy",
        "MM/dd/yyyy",
        "yyyy-MM-dd",
        "M/d/yyyy",
        "M/dd/yyyy",
        "MM/dd/yy",
        "MM.dd.yyyy",
        "dd.MM.yyyy",
        "yyyy-MM-dd",
        "yyyy-dd-MM",
    )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

from datetime import datetime
now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_1_df = output_1_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_1_df = output_1_df.withColumn("date", F.lit(date))
output_1_df = output_1_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in output_1_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  output_1_df = output_1_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none": 
  spark.sql(f"USE CATALOG {catalog_name}")
else:
  spark.sql(f"USE CATALOG hive_metastore")

# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

output_1_df.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

if catalog_name and catalog_name.lower() != "none": 
  output_1_table_path = output_path
else:
  output_1_table_path = spark.sql(f"desc {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Hive Path : {output_1_table_path}")

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### REGISTER THE FEATURES ON MLCORE
# MAGIC

# COMMAND ----------

# DBTITLE 1,Register Features Transformed Table
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = output_table_configs["output_1"]["table"],
    num_rows = output_1_df.count(),
    cols = output_1_df.columns,
    column_datatype = output_1_df.dtypes,
    table_schema = output_1_df.schema,
    primary_keys = output_table_configs["output_1"]["primary_keys"],
    table_path = output_1_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal" ,
    table_sub_type="Source",
    request_type = task,
    tracking_env = env,
    batch_size = str(batch_size),
    quartz_cron_expression = cron_job_schedule,
    compute_usage_metrics = compute_metrics,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    verbose = True,
    input_table_names = input_table_paths['input_1'],
    )

# COMMAND ----------

if is_scheduled:
  obj_properties['end_marker'] = end_marker
  with open(pickle_file_path, "wb") as handle:
      pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Instance successfully saved successfully")

# COMMAND ----------

# try :
#     #define media artifacts path
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils,
#         env = env)
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Custom_EDA", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#             "input_table_path" : features_hive_table_path,
#             "media_artifacts_path" : media_artifacts_path,
#             })
# except Exception as e:
#     print(f"Exception while triggering EDA notebook : {e}")

# COMMAND ----------


