# Databricks notebook source
general_configs : 
  sdk_session_id: 
    dev: 5a5d195c7715450ba656271aa842b054 #9f406b2dfe6d4bb1b52ec796e653ae79
    uat: 16ba37d0994549a7a3fb1df72563e37c
    prod: 64ec92d6117247877965d152
  tracking_env : dev

data_engineering_ft:
  datalake_configs:
    input_tables :
      source_1 : 
        catalog_name : mlcore_dev
        schema : windforecast_unitycatalog_1
        table : raw_wind_forecast_feature_1
        primary_keys: index
    output_tables :
      output_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2506_v1 #wind_forecast_uc_db_2106_v1
        table: wind_forecast_feature_uc_2106_v1
        primary_keys: index
  storage_configs :
    cloud_provider : databricks_uc
    params:
      metastore : null
      catalog_name : mlcore_dev
      schema_name : mlcore
      volume_name : mlcore_volume
  is_scheduled : true
  batch_size : 500
  cron_job_schedule: 0 */30 * ? * *

data_engineering_gt:
  datalake_configs:
    input_tables :
      source_1 : 
        catalog_name : mlcore_dev
        schema : windforecast_unitycatalog_1
        table : raw_wind_forecast_gt_1
        primary_keys: index
    output_tables :
      output_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: wind_forecast_gt_uc_2106_v1
        primary_keys: index
  storage_configs :
    cloud_provider : databricks_uc
    params:
      metastore : null
      catalog_name : mlcore_dev
      schema_name : mlcore
      volume_name : mlcore_volume
  is_scheduled : true
  batch_size : 500
  cron_job_schedule: 0 */30 * ? * *

feature_pipelines_ft:
  datalake_configs:
    input_tables :
      input_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: wind_forecast_feature_uc_2106_v1
        primary_keys: index
    output_tables :
        output_1 :
          catalog_name : mlcore_dev
          schema : wind_forecast_uc_db_2106_v1
          table: transformed_wind_forecast_feature_uc_2106_v1
          primary_keys: index
  storage_configs :
    cloud_provider : databricks_uc
    params:
      metastore : null
      catalog_name : mlcore_dev
      schema_name : mlcore
      volume_name : mlcore_volume
  is_scheduled: false

feature_pipelines_gt:
  datalake_configs:
    input_tables : 
      input_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: wind_forecast_gt_uc_2106_v1
        primary_keys: index
    output_tables : 
        output_1 :
          catalog_name : mlcore_dev
          schema : wind_forecast_uc_db_2106_v1
          table: transformed_wind_forecast_gt_uc_2106_v1
          primary_keys: index  
  storage_configs :
    cloud_provider : databricks_uc
    params:
      metastore : null
      catalog_name : mlcore_dev
      schema_name : mlcore
      volume_name : mlcore_volume
  is_scheduled: false

train:
  datalake_configs:
    input_tables : 
      input_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: transformed_wind_forecast_feature_uc_2106_v1
        primary_keys: index
      input_2 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: transformed_wind_forecast_gt_uc_2106_v1
        primary_keys: index
    output_tables : 
        output_1 :
          catalog_name : mlcore_dev
          schema : wind_forecast_uc_db_2106_v1
          table: trainoutput_wind_forecast_uc_2106_v1
          primary_keys: index
  model_configs :
    registry_platform: databricks
    model_registry : mlflow
    unity_catalog : "yes"
    model_registry_params :
      catalog_name: mlcore_dev
      schema_name: wind_forecast_uc_db_2106_v1
    model_params:      
      model_name: wind_forecast_az_mlflow_2106_v1
  storage_configs :
    cloud_provider : databricks_uc
    params:
      metastore : null
      catalog_name : mlcore_dev
      schema_name : mlcore
      volume_name : mlcore_volume
  test_size : 0.2
  feature_columns:
    - AmbientTemperatue
    - BearingShaftTemperature
    - Blade1PitchAngle
    - Blade2PitchAngle
    - Blade3PitchAngle
    - GearboxBearingTemperature
    - GearboxOilTemperature
    - GeneratorRPM
    - GeneratorWinding1Temperature
    - GeneratorWinding2Temperature
    - HubTemperature
    - MainBoxTemperature
    - NacellePosition
    - ReactivePower
    - RotorRPM
    - TurbineStatus
    - WindDirection
    - WindSpeed
    - day_of_week
    - quarter
  date_column: 'date_time'
  target_columns:
  - ActivePower
  horizon: 7
  frequency: 'D'
  is_scheduled: false

data_prep_deployment_ft:
  datalake_configs:
    input_tables :
      input_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: wind_forecast_feature_uc_2106_v1
        primary_keys: index
    output_tables :
        output_1 :
          catalog_name : mlcore_dev
          schema : wind_forecast_uc_db_2106_v1
          table: dpd_wind_forecast_feature_uc_2106_v1
          primary_keys: index  
  storage_configs :
    cloud_provider : databricks_uc
    params:
      metastore : null
      catalog_name : mlcore_dev
      schema_name : mlcore
      volume_name : mlcore_volume
  is_scheduled: true
  batch_size: 100
  cron_job_schedule: 0 */30 * ? * *

data_prep_deployment_gt:
  datalake_configs:
    input_tables : 
      input_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: wind_forecast_gt_uc_2106_v1
        primary_keys: index
    output_tables : 
        output_1 :
          catalog_name : mlcore_dev
          schema : wind_forecast_uc_db_2106_v1
          table: dpd_wind_forecast_gt_uc_2106_v1
          primary_keys: index  
  storage_configs :
    cloud_provider : databricks_uc
    params:
      metastore : null
      catalog_name : mlcore_dev
      schema_name : mlcore
      volume_name : mlcore_volume
  is_scheduled: true
  batch_size: 100
  cron_job_schedule: 0 */30 * ? * *

inference:
  datalake_configs:
    input_tables : 
      input_1 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: dpd_wind_forecast_feature_uc_2106_v1
        primary_keys: index
      input_2 :
        catalog_name : mlcore_dev
        schema : wind_forecast_uc_db_2106_v1
        table: dpd_wind_forecast_gt_uc_2106_v1
        primary_keys: index
    output_tables : 
        output_1 :
          catalog_name : mlcore_dev
          schema : wind_forecast_uc_db_2106_v1
          table: inference_wind_forecast_2106_v1
          primary_keys: index
  model_configs :
    use_latest_version: true
  is_scheduled: true
  batch_size: 100
  cron_job_schedule: 0 */30 * ? * *
