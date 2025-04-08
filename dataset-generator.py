import pandas as pd
import numpy as np
import random

# Define data for each column
def generate_customer_churn_data(num_customers=1000):
    customerID = [f"CUST{str(i).zfill(5)}" for i in range(1, num_customers + 1)]
    gender = [random.choice(["Male", "Female"]) for _ in range(num_customers)]
    
    # Senior citizens have a higher chance to churn but not overly so
    senior_citizen = [random.choices([0, 1], weights=[0.8, 0.2])[0] for _ in range(num_customers)]
    
    # Tenure distribution: more short-tenure and long-tenure customers
    tenure = [random.randint(0, 12) if random.random() < 0.4 else random.randint(12, 72) for _ in range(num_customers)]
    
    # Monthly charges with more overlap
    monthly_charges = [round(random.uniform(60, 110) if tenure[i] < 12 else random.uniform(20, 100), 2) for i in range(num_customers)]
    
    # Add some noise for TotalCharges based on MonthlyCharges and Tenure
    total_charges = [
        round(monthly_charges[i] * tenure[i] + random.uniform(-10, 10) * tenure[i] if tenure[i] > 0 else np.nan, 2) 
        for i in range(num_customers)
    ]
    
    # Phone and Internet Service with mixed churn rates
    phone_service = [random.choice(["Yes", "No"]) for _ in range(num_customers)]
    internet_service = [
        random.choices(["DSL", "Fiber optic", "No"], weights=[0.4, 0.4, 0.2])[0] for _ in range(num_customers)
    ]
    
    # Generate churn with less determinism and more complex conditions
    churn = [
        "Yes" if (
            (tenure[i] < 12 and monthly_charges[i] > 70) or
            (senior_citizen[i] == 1 and random.random() < 0.4) or
            (internet_service[i] == "Fiber optic" and random.random() < 0.5)
        ) and random.random() < 0.6  # Random noise for churn
        else "No"
        for i in range(num_customers)
    ]

    # Compile the data into a DataFrame
    data = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    }
    df = pd.DataFrame(data)

    # Introduce some NaN values randomly in 'TotalCharges' column
    nan_indices = np.random.choice(df.index, size=int(num_customers * 0.05), replace=False)
    df.loc[nan_indices, 'TotalCharges'] = np.nan

    return df

# Generate the dataset
df = generate_customer_churn_data(num_customers=1000)
df.to_csv("./customer_churn.csv", index=False)
