from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import when, col

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset (ensure "customer_churn.csv" is in your working directory)
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# ----------------------------
# Task 1: Data Preprocessing and Feature Engineering
# ----------------------------
def preprocess_data(df):
    """
    Objective:
    Clean the dataset and prepare features for ML algorithms.
    
    Steps:
      1. Fill missing values in TotalCharges with 0.
      2. Convert categorical variables and the label 'Churn'.
         - Map Churn values ("Yes"/"No") to a numeric column 'ChurnIndex'.
      3. Encode categorical features using StringIndexer and OneHotEncoder.
      4. Assemble numeric and encoded features into a single feature vector with VectorAssembler.
    """
    # 1. Fill missing values in TotalCharges with 0.
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))
    
    # 2. Convert Churn values: "Yes" -> 1, "No" -> 0, stored in a new column "ChurnIndex"
    df = df.withColumn("ChurnIndex", when(col("Churn") == "Yes", 1).otherwise(0))
    
    # 3. Encode categorical features. Here, we assume these three columns: gender, PhoneService, InternetService.
    categorical_cols = ["gender", "PhoneService", "InternetService"]
    for c in categorical_cols:
        # Use StringIndexer to convert string labels to numeric indices.
        indexer = StringIndexer(inputCol=c, outputCol=c + "_Index", handleInvalid="keep")
        df = indexer.fit(df).transform(df)
        # One-hot encode the indexed column.
        encoder = OneHotEncoder(inputCols=[c + "_Index"], outputCols=[c + "_OHE"])
        df = encoder.fit(df).transform(df)
    
    # 4. Assemble features.
    # Define numerical features — adjust these if your dataset contains other numeric columns.
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    encoded_features = [c + "_OHE" for c in categorical_cols]
    feature_columns = numeric_features + encoded_features
    
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    final_df = assembler.transform(df)
    
    # Select only the "features" vector and the numeric label "ChurnIndex" for the ML pipeline.
    final_df = final_df.select("features", "ChurnIndex")
    
    return final_df

# ----------------------------
# Task 2: Train and Evaluate Logistic Regression Model
# ----------------------------
def train_logistic_regression_model(df):
    """
    Objective:
    Train a logistic regression model and evaluate it using AUC.
    
    Steps:
      1. Split dataset into training and test sets (80/20).
      2. Train a logistic regression model using the training data.
      3. Use BinaryClassificationEvaluator to evaluate the test set performance.
      
    Sample Code Output:
    Logistic Regression Model Accuracy (AUC): 0.83
    """
    # Split the data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize and train Logistic Regression.
    lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndex")
    lr_model = lr.fit(train_df)
    
    # Make predictions on the test set.
    predictions = lr_model.transform(test_df)
    
    # Evaluate using AUC.
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndex", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("Logistic Regression Model Accuracy (AUC): {:.2f}".format(auc))
    
# ----------------------------
# Task 3: Feature Selection using Chi-Square Test
# ----------------------------
def feature_selection(df):
    """
    Objective:
    Select the top 5 most important features using Chi-Square feature selection.
    
    Steps:
      1. Use ChiSqSelector to rank and select the top 5 features.
      2. Print the selected feature vector and the label column.
      
    Sample Code Output:
    +-------------------------------+-----------+
    |selectedFeatures               |ChurnIndex |
    +-------------------------------+-----------+
    |[0.0,29.85,0.0,0.0, ...]        |0          |
    |[1.0,56.95,1.0,0.0, ...]        |1          |
    |...                            |...        |
    +-------------------------------+-----------+
    """
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="ChurnIndex", outputCol="selectedFeatures")
    selected_df = selector.fit(df).transform(df)
    
    print("Selected features after ChiSqSelector:")
    selected_df.select("selectedFeatures", "ChurnIndex").show(5, truncate=False)
    
# ----------------------------
# Task 4: Hyperparameter Tuning and Model Comparison
# ----------------------------
def tune_and_compare_models(df):
    """
    Objective:
    Use CrossValidator to tune models and compare their AUC performance.
    
    Models Used:
      - Logistic Regression
      - Decision Tree Classifier
      - Random Forest Classifier
      - Gradient Boosted Trees (GBT)
      
    Steps:
      1. Define models and their corresponding hyperparameter grids.
      2. Use 5-fold CrossValidator for hyperparameter tuning.
      3. Evaluate and print each model's best AUC and hyperparameters.
      
    Sample Code Output Example:
      Tuning LogisticRegression...
      LogisticRegression Best Model Accuracy (AUC): 0.84
      Best Params for LogisticRegression: regParam=0.01, maxIter=20
      ...
    """
    # Split the data.
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndex", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    
    # Save model results
    model_results = []
    
    # Logistic Regression
    print("Tuning LogisticRegression...")
    lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndex")
    lr_paramGrid = ParamGridBuilder() \
                    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
                    .addGrid(lr.maxIter, [10, 20]) \
                    .build()
    lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_paramGrid, evaluator=evaluator, numFolds=5)
    lr_cvModel = lr_cv.fit(train_df)
    lr_predictions = lr_cvModel.transform(test_df)
    lr_auc = evaluator.evaluate(lr_predictions)
    print("LogisticRegression Best Model Accuracy (AUC): {:.2f}".format(lr_auc))
    print("Best Params for LogisticRegression:", lr_cvModel.bestModel.extractParamMap())
    model_results.append(("LogisticRegression", lr_auc, lr_cvModel.bestModel.extractParamMap()))
    
    # Decision Tree Classifier
    print("\nTuning DecisionTree...")
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="ChurnIndex")
    dt_paramGrid = ParamGridBuilder() \
                    .addGrid(dt.maxDepth, [5, 10]) \
                    .addGrid(dt.minInstancesPerNode, [1, 2]) \
                    .build()
    dt_cv = CrossValidator(estimator=dt, estimatorParamMaps=dt_paramGrid, evaluator=evaluator, numFolds=5)
    dt_cvModel = dt_cv.fit(train_df)
    dt_predictions = dt_cvModel.transform(test_df)
    dt_auc = evaluator.evaluate(dt_predictions)
    print("DecisionTree Best Model Accuracy (AUC): {:.2f}".format(dt_auc))
    print("Best Params for DecisionTree:", dt_cvModel.bestModel.extractParamMap())
    model_results.append(("DecisionTree", dt_auc, dt_cvModel.bestModel.extractParamMap()))
    
    # Random Forest Classifier
    print("\nTuning RandomForest...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="ChurnIndex")
    rf_paramGrid = ParamGridBuilder() \
                    .addGrid(rf.numTrees, [10, 20, 50]) \
                    .addGrid(rf.maxDepth, [5, 10, 15]) \
                    .build()
    rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_paramGrid, evaluator=evaluator, numFolds=5)
    rf_cvModel = rf_cv.fit(train_df)
    rf_predictions = rf_cvModel.transform(test_df)
    rf_auc = evaluator.evaluate(rf_predictions)
    print("RandomForest Best Model Accuracy (AUC): {:.2f}".format(rf_auc))
    print("Best Params for RandomForest:", rf_cvModel.bestModel.extractParamMap())
    model_results.append(("RandomForest", rf_auc, rf_cvModel.bestModel.extractParamMap()))
    
    # Gradient Boosted Trees (GBT) Classifier
    print("\nTuning GBT...")
    gbt = GBTClassifier(featuresCol="features", labelCol="ChurnIndex")
    gbt_paramGrid = ParamGridBuilder() \
                    .addGrid(gbt.maxIter, [10, 20]) \
                    .addGrid(gbt.maxDepth, [3, 5, 10]) \
                    .build()
    gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_paramGrid, evaluator=evaluator, numFolds=5)
    gbt_cvModel = gbt_cv.fit(train_df)
    gbt_predictions = gbt_cvModel.transform(test_df)
    gbt_auc = evaluator.evaluate(gbt_predictions)
    print("GBT Best Model Accuracy (AUC): {:.2f}".format(gbt_auc))
    print("Best Params for GBT:", gbt_cvModel.bestModel.extractParamMap())
    model_results.append(("GBT", gbt_auc, gbt_cvModel.bestModel.extractParamMap()))
    
    return model_results

# ----------------------------
# Execute the tasks
# ----------------------------

# Task 1: Data Preprocessing and Feature Engineering
preprocessed_df = preprocess_data(df)
print("Task 1: Data Preprocessing and Feature Engineering")
print("Objective:")
print("Clean the dataset and prepare features for ML algorithms.\n")
print("Steps:")
print("  1. Fill missing values in TotalCharges with 0.")
print("  2. Encode categorical features using StringIndexer and OneHotEncoder.")
print("  3. Assemble numeric and encoded features into a single feature vector with VectorAssembler.\n")
print("Code Output:")
preprocessed_df.select("features", "ChurnIndex").show(5, truncate=False)

# Task 2: Train and Evaluate Logistic Regression Model
print("\nTask 2: Train and Evaluate Logistic Regression Model")
print("Objective:")
print("Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).\n")
print("Steps:")
print("  1. Split dataset into training and test sets (80/20).")
print("  2. Train a logistic regression model.")
print("  3. Use BinaryClassificationEvaluator to evaluate.\n")
print("Code Output Example:")
train_logistic_regression_model(preprocessed_df)

# Task 3: Feature Selection using Chi-Square Test
print("\nTask 3: Feature Selection using Chi-Square Test")
print("Objective:")
print("Select the top 5 most important features using Chi-Square feature selection.\n")
print("Steps:")
print("  1. Use ChiSqSelector to rank and select top 5 features.")
print("  2. Print the selected feature vectors.\n")
print("Code Output Example:")
feature_selection(preprocessed_df)

# Task 4: Hyperparameter Tuning and Model Comparison
print("\nTask 4: Hyperparameter Tuning and Model Comparison")
print("Objective:")
print("Use CrossValidator to tune models and compare their AUC performance.\n")
print("Models Used:")
print("  - Logistic Regression")
print("  - Decision Tree Classifier")
print("  - Random Forest Classifier")
print("  - Gradient Boosted Trees (GBT)\n")
print("Steps:")
print("  1. Define models and parameter grids.")
print("  2. Use CrossValidator for 5-fold cross-validation.")
print("  3. Evaluate and print best model results.\n")
print("Code Output Example:")
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
