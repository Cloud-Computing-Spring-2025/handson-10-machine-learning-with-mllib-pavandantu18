from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import when, col

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
from pyspark.sql.functions import when, col

def preprocess_data(df):
    """
    Prepares the raw data by:
    - Filling missing values,
    - Converting the Churn column (string) to numeric,
    - Encoding categorical variables,
    - One-hot encoding the indexed features,
    - Assembling all features into a single feature vector.
    """
    # Fill missing values in TotalCharges column with 0
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))
    
    # Convert the 'Churn' column from string to numeric (assuming "Yes"->1, "No"->0)
    df = df.withColumn("Churn", when(col("Churn") == "Yes", 1).otherwise(0))
    
    # Define the categorical columns to encode
    categorical_cols = ["gender", "PhoneService", "InternetService"]
    
    # Convert categorical columns to indexed numeric values and then one-hot encode them.
    for col_name in categorical_cols:
        # Indexing categorical column (handling unseen values with "keep")
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_Index", handleInvalid="keep")
        df = indexer.fit(df).transform(df)
        
        # One-hot encoding the indexed column
        encoder = OneHotEncoder(inputCols=[col_name + "_Index"], outputCols=[col_name + "_OHE"])
        df = encoder.fit(df).transform(df)
    
    # Assemble features into a single vector.
    # Assumed numerical features: SeniorCitizen, tenure, MonthlyCharges, TotalCharges.
    feature_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + [c + "_OHE" for c in categorical_cols]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    final_df = assembler.transform(df)
    
    return final_df


# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    """
    Splits the preprocessed data into training and testing sets, trains a logistic regression model,
    and evaluates it using the AUC metric.
    
    Assumes that the 'Churn' column is the label (0 for no churn, 1 for churn).
    """
    # Split into training (80%) and testing (20%) sets.
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Define and train the Logistic Regression model.
    lr = LogisticRegression(featuresCol="features", labelCol="Churn", maxIter=10)
    lr_model = lr.fit(train_df)
    
    # Make predictions on the test set.
    predictions = lr_model.transform(test_df)
    
    # Evaluate the model using BinaryClassificationEvaluator with AUC metric.
    evaluator = BinaryClassificationEvaluator(labelCol="Churn", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("Logistic Regression AUC: {:.4f}".format(auc))

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    """
    Uses a Chi-Square selector to reduce the feature vector down to the top 5 features most relevant
    to predicting churn, and then displays a sample of the selected features along with the label column.
    """
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="Churn", outputCol="selectedFeatures")
    result = selector.fit(df).transform(df)
    
    print("Sample of selected features (top 5) and label:")
    result.select("selectedFeatures", "Churn").show(5, truncate=False)

# Task 4: Hyperparameter Tuning and Model Comparison with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    """
    Splits the data, defines multiple classification models along with their hyperparameter grids,
    and uses 5-fold cross-validation to find and compare the best model based on AUC.
    """
    # Split into training (80%) and testing (20%) sets.
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Define the evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="Churn", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    
    best_auc = 0
    best_model_details = {}
    
    # List to hold (model name, model, hyperparameter grid) tuples.
    models = []
    
    # Logistic Regression model and hyperparameter grid.
    lr = LogisticRegression(featuresCol="features", labelCol="Churn")
    lr_paramGrid = ParamGridBuilder() \
                    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
                    .addGrid(lr.maxIter, [10, 20]) \
                    .build()
    models.append(("LogisticRegression", lr, lr_paramGrid))
    
    # Decision Tree Classifier and hyperparameter grid.
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="Churn")
    dt_paramGrid = ParamGridBuilder() \
                    .addGrid(dt.maxDepth, [5, 10]) \
                    .addGrid(dt.minInstancesPerNode, [1, 2]) \
                    .build()
    models.append(("DecisionTree", dt, dt_paramGrid))
    
    # Random Forest Classifier and hyperparameter grid.
    rf = RandomForestClassifier(featuresCol="features", labelCol="Churn")
    rf_paramGrid = ParamGridBuilder() \
                    .addGrid(rf.numTrees, [10, 20]) \
                    .addGrid(rf.maxDepth, [5, 10]) \
                    .build()
    models.append(("RandomForest", rf, rf_paramGrid))
    
    # Gradient Boosted Trees (GBT) Classifier and hyperparameter grid.
    gbt = GBTClassifier(featuresCol="features", labelCol="Churn")
    gbt_paramGrid = ParamGridBuilder() \
                    .addGrid(gbt.maxIter, [10, 20]) \
                    .addGrid(gbt.maxDepth, [3, 5]) \
                    .build()
    models.append(("GBT", gbt, gbt_paramGrid))
    
    # Loop through the models and perform cross-validation.
    for model_name, model, paramGrid in models:
        print(f"Starting cross-validation for {model_name}...")
        cv = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
        cvModel = cv.fit(train_df)
        predictions = cvModel.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"{model_name} AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model_details = {
                "model_name": model_name,
                "bestModel": cvModel.bestModel,
                "bestParams": cvModel.bestModel.extractParamMap(),
                "auc": best_auc
            }
    
    # Report the best model's details.
    print("\nBest Model Details:")
    print("Model:", best_model_details.get("model_name"))
    print("AUC:", best_model_details.get("auc"))
    print("Best Hyperparameters:", best_model_details.get("bestParams"))
    
# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop the Spark session
spark.stop()
