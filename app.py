import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


df = pd.read_csv('titanic.csv')

# Title and description
st.title("Titanic Dataset Exploration")
st.write("This app provides basic visualizations for the Titanic dataset.")

# Show dataset
if st.checkbox("Show Raw Dataset"):
    st.write(df)

# Sidebar for user input
st.sidebar.header("Filter Options")
sex_filter = st.sidebar.multiselect("Select Gender", options=df['sex'].unique(), default=df['sex'].unique())
class_filter = st.sidebar.multiselect("Select Class (Pclass)", options=df['pclass'].unique(), default=df['pclass'].unique())
embarked_filter = st.sidebar.multiselect("Select Embarkation Point", options=df['embark_town'].dropna().unique(), default=df['embark_town'].unique())

# Filter dataset based on selections
filtered_df = df[
    (df['sex'].isin(sex_filter)) & 
    (df['pclass'].isin(class_filter)) & 
    (df['embark_town'].isin(embarked_filter))
]

# Display filtered dataset
st.write("### Filtered Dataset")
st.write(filtered_df)

# Viz 1: Survival countplot
st.subheader("Survival Count")
survival_count = filtered_df['survived'].value_counts()
st.bar_chart(survival_count.rename(index={0: 'Did not survive', 1: 'Survived'}))

# Viz 2: Survival by gender
st.subheader("Survival Rate by Gender")

gender_survival = filtered_df.groupby('sex')['survived'].mean().reset_index()

fig = px.pie(
    gender_survival,
    names='sex',
    values='survived',
    title="Survival Rate by Gender",
)

# Display the pie chart
st.plotly_chart(fig)

# Viz 3: Age distribution
# Prepare data for histogram
age_bins = pd.cut(filtered_df['age'].dropna(), bins=20)  # Create 20 bins for ages
hist_data = age_bins.value_counts(sort=False)  # Count ages in each bin

# Convert to DataFrame for plotting
hist_df = pd.DataFrame({
    'Age Range': [f"{int(bin.left)}-{int(bin.right)}" for bin in hist_data.index],  # Bin ranges as labels
    'Count': hist_data.values
})

# Display histogram using Streamlit
st.bar_chart(hist_df.set_index('Age Range'))
