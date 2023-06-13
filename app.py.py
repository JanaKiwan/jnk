

import streamlit as st


import pandas as pd
from pandas_profiling import ProfileReport as PR
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import gc

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn.impute import KNNImputer

import joblib

from imblearn.over_sampling import SMOTENC


import warnings
warnings.filterwarnings("ignore")




data=pd.read_csv("healthcare-dataset-stroke-data.csv")
data = data.loc[data.gender != "Other"]



#Bar Chart - gender distribution
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Count the occurrences of each gender
gender_counts = data['gender'].value_counts()

# Create a bar chart
plt.bar(gender_counts.index, gender_counts.values)

# Customize the plot
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')

# Display the plot using Streamlit
st.pyplot()




# Create a histogram of age distribution
plt.hist(data['age'], bins=10)

# Customize the plot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')

# Display the plot using Streamlit
st.pyplot()




# Count the occurrences of hypertension
hypertension_counts = data['hypertension'].value_counts()

# Create a pie chart
plt.pie(hypertension_counts, labels=hypertension_counts.index, autopct='%1.1f%%')

# Customize the plot
plt.title('Hypertension Distribution')

# Display the plot using Streamlit
st.pyplot()




# Group the data by age and calculate average glucose level
age_glucose = data.groupby('age')['avg_glucose_level'].mean().reset_index()

# Create a line plot
sns.lineplot(x='age', y='avg_glucose_level', data=age_glucose)

# Customize the plot
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.title('Average Glucose Level by Age')

# Display the plot using Streamlit
st.pyplot()




# Create a box plot
sns.boxplot(x='work_type', y='bmi', data=data)

# Customize the plot
plt.xlabel('Work Type')
plt.ylabel('BMI')
plt.title('BMI Distribution by Work Type')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Display the plot using Streamlit
st.pyplot()




# Create a scatter plot
plt.scatter(data['age'], data['avg_glucose_level'])

# Customize the plot
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.title('Age vs. Average Glucose Level')

# Display the plot using Streamlit
st.pyplot()



import plotly.express as px
# Create a stacked bar chart
stacked_bar_chart = px.bar(data, x='smoking_status', color='heart_disease', title='Heart Disease Count by Smoking Status')
stacked_bar_chart.update_layout(barmode='stack')

# Display the chart using Streamlit
st.plotly_chart(stacked_bar_chart)




# Create a violin plot
sns.violinplot(x='stroke', y='bmi', data=data)

# Customize the plot
plt.xlabel('Stroke')
plt.ylabel('BMI')
plt.title('BMI Distribution for Individuals with and without Stroke')

# Display the plot using Streamlit
st.pyplot()




#correlation between age, average glucose level, and BMI using Seaborn
# Select the columns for correlation
corr_df = data[['age', 'avg_glucose_level', 'bmi']]

# Calculate the correlation matrix
corr_matrix = corr_df.corr()

# Create a heatmap
sns.heatmap(corr_matrix, annot=True)

# Customize the plot
plt.title('Correlation Heatmap')

# Display the plot using Streamlit
st.pyplot()



# Count the occurrences of marital status
marital_counts = data['ever_married'].value_counts()

# Create a donut chart
fig = go.Figure(data=[go.Pie(labels=marital_counts.index, values=marital_counts.values, hole=.3)])

# Customize the chart
fig.update_layout(title_text='Marital Status Distribution')

# Display the chart using Streamlit
st.plotly_chart(fig)




# Categorical variables for analysis
catVars = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# Function to calculate stroke proportions
def strokeProportion(data, column):
    grouped = data.groupby(column)["stroke"].value_counts().unstack()
    total = grouped.sum(axis=1)
    proportions = grouped[1] / total
    return proportions

# Streamlit app
def main():
    st.title("Proportion of Strokes in Categorical Variables")

    # Display the proportion of strokes for each categorical variable
    for var in catVars:
        proportions = strokeProportion(data, var)
        st.subheader(f"Proportion of Strokes by {var}")
        st.bar_chart(proportions)

if __name__ == "__main__":
    main()







