# Databricks notebook source
import yaml

# COMMAND ----------

with open('SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config) 

# COMMAND ----------


