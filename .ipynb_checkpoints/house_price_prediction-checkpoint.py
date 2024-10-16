import pandas as pd

# pd.read_csv to load .csv files
df = pd.read_excel("HousePricePrediction.xlsx")

# Step 1. Data Processing (expand to see note)
# Usually done with:
# describe, dtypes, head/tail, shape
'''
We categorize the data for few reasons:
    - Data Preprocessing: Applying the right transformations (e.g., scaling for numerical, encoding for categorical).
    - Statistical Analysis: Using appropriate statistical measures (e.g., mean for numerical, frequency for categorical).
    - Handling Missing Data: Different strategies for filling missing values depending on datatype.
    - Model Compatibility: Ensuring data is in the correct format for the chosen machine learning model.
    - Memory Efficiency: Optimizing memory usage with appropriate data types.
    - Data Visualization: Choosing the right visualization technique for each datatype.
    - Feature Engineering: Enabling better feature transformations based on datatype.
'''

# print(df.describe)
# print(df.head(5))
# print(df.tail(5))
# print(df.shape)

# returns True if dtype == 'object', else, False
# repeat for int and float
objects = (df.dtypes == 'object')
objectsList = list(objects[objects].index) # Filter out all the False and convert the index to a list
# print('Number of Objects: ', str(len(objectsList)))

ints = (df.dtypes == 'int')
intsList = list(ints[ints].index) # Filter out all the False and convert the index to a list
# print('Number of Integer: ', str(len(intsList)))

floats = (df.dtypes == 'float')
floatsList = list(floats[floats].index) # Filter out all the False and convert the index to a list
# print('Number of Float: ', str(len(floatsList)))
