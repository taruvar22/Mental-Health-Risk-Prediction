# ============================================
# 🧠 Mental Health Risk Prediction (PySpark)
# ============================================

# Import Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# 1. Initialize Spark Session
# ============================================
spark = SparkSession.builder \
    .appName("MentalHealthRiskPrediction") \
    .getOrCreate()

print("✅ Spark Session Started")

# ============================================
# 2. Load Dataset
# ============================================
file_path = "/Users/taruvar/Downloads/cleaned_dataset.csv"

df = spark.read.csv(file_path, header=True, inferSchema=True)

print("✅ Dataset Loaded")
df.show(5)
df.printSchema()

# ============================================
# 3. Data Cleaning
# ============================================
# Drop missing values
df = df.dropna()

# Normalize Gender column
df = df.withColumn("Gender",
    when(col("Gender").isin("Male", "male", "M"), "Male")
    .when(col("Gender").isin("Female", "female", "F"), "Female")
    .otherwise("Other")
)

print("✅ Data Cleaned")
df.select("Age", "Gender").show(5)

# ============================================
# 4. Encode Target Variable
# ============================================
label_indexer = StringIndexer(inputCol="target", outputCol="label")

# ============================================
# 5. Encode Categorical Features
# ============================================
categorical_cols = ["Gender", "family_history", "remote_work", "no_employees"]

indexers = [
    StringIndexer(inputCol=col_name, outputCol=col_name + "_idx")
    for col_name in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=col_name + "_idx", outputCol=col_name + "_vec")
    for col_name in categorical_cols
]

# ============================================
# 6. Feature Vector Creation
# ============================================
assembler = VectorAssembler(
    inputCols=["Age"] + [col_name + "_vec" for col_name in categorical_cols],
    outputCol="features"
)

# ============================================
# 7. Model Definition
# ============================================
lr = LogisticRegression(featuresCol="features", labelCol="label")

# ============================================
# 8. Create Pipeline
# ============================================
pipeline = Pipeline(
    stages=[label_indexer] + indexers + encoders + [assembler, lr]
)

# ============================================
# 9. Train-Test Split
# ============================================
train, test = df.randomSplit([0.8, 0.2], seed=42)

print("✅ Data Split Completed")

# ============================================
# 10. Train Model
# ============================================
model = pipeline.fit(train)

print("✅ Model Training Completed")

# ============================================
# 11. Predictions
# ============================================
predictions = model.transform(test)

print("✅ Predictions Generated")
predictions.select("features", "label", "prediction").show(5)

# ============================================
# 12. Model Evaluation
# ============================================
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

print("===================================")
print("📊 Model Evaluation Results")
print("Accuracy:", accuracy)
print("F1 Score:", f1_score)
print("===================================")

# ============================================
# 13. Visualization
# ============================================
print("📊 Generating Visualization...")

pdf = predictions.select("Gender", "prediction").toPandas()

sns.countplot(x="Gender", hue="prediction", data=pdf)
plt.title("Mental Health Risk Prediction by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ============================================
# 14. Stop Spark Session
# ============================================
spark.stop()

print("✅ Spark Session Stopped")
