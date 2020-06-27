import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr

# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")

def show_dtypes(x):
    return x.dtypes

def show_columns(x):
    return x.columns

def Show_Missing(x):
    return x.isna().sum()
def Show_Missing1(x):
    return x.isna().sum()

from scipy import stats
def Tabulation(x):
    table = pd.DataFrame(x.dtypes,columns=['dtypes'])
    table1 =pd.DataFrame(x.columns,columns=['Names'])
    table = table.reset_index()
    table= table.rename(columns={'index':'Name'})
    table['No of Missing'] = x.isnull().sum().values    
    table['No of Uniques'] = x.nunique().values
    table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
    table['First Observation'] = x.loc[0].values
    table['Second Observation'] = x.loc[1].values
    table['Third Observation'] = x.loc[2].values
    for name in table['Name'].value_counts().index:
        table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2)
    return table


def Numerical_variables(x):
    Num_var = [var for var in x.columns if x[var].dtypes!="object"]
    Num_var = x[Num_var]
    return Num_var

def categorical_variables(x):
    cat_var = [var for var in x.columns if x[var].dtypes=="object"]
    cat_var = x[cat_var]
    return cat_var

def impute(x):
    df=x.dropna()
    return df

def Show_pearsonr(x,y):
    result = pearsonr(x,y)
    return result

from scipy.stats import spearmanr
def Show_spearmanr(x,y):
    result = spearmanr(x,y)
    return result

import plotly.express as px
def plotly(a,x,y):
    fig = px.scatter(a, x=x, y=y)
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def plotly_histogram(a,x,y):
    fig = px.histogram(a, x=x, y=y)
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
    
    
def main():
	st.title("Machine Learning Application for Automated EDA")
	html_temp = """
	<div style="background-color:tomato;padding:10px">
	<h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
	</div>
	"""
	"""By DHEERAJ KUMAR"""
	"""https://github.com/DheerajKumar97""" 
	activities = ["EDA","Plots"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			

			if st.checkbox("Show dtypes"):
				st.write(show_dtypes(df))

			if st.checkbox("Aggregation Tabulation"):
				st.write(Tabulation(df))

			if st.checkbox("Show Columns"):
				st.write(show_columns(df))

			if st.checkbox("Show Missing"):
				st.write(Show_Missing1(df))

# 			if st.checkbox("Show Selected Columns"):
# 				selected_columns = st.multiselect("Select Columns",all_columns)
# 				new_df = df[selected_columns]
# 				st.dataframe(new_df)

                
			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",show_columns(df))
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Num Variables"):
				num_df = Numerical_variables(df)
				numer_df=pd.DataFrame(num_df)                
				st.dataframe(numer_df)

			if st.checkbox("Cat Variables"):
				new_df = categorical_variables(df)
				catego_df=pd.DataFrame(new_df)                
				st.dataframe(catego_df)

			if st.checkbox("DropNA"):
				imp_df = impute(num_df)
				st.dataframe(imp_df)


			if st.checkbox("Missing after DropNA"):
				st.write(Show_Missing(imp_df))
               

			all_columns_names = show_columns(df)
			all_columns_names1 = show_columns(df)            
			selected_columns_names = st.selectbox("Select Columns Cross Tb",all_columns_names)
			selected_columns_names1 = st.selectbox("Select Columns Cross",all_columns_names1)
			if st.button("Generate Cross Tab"):
				st.dataframe(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))


			all_columns_names3 = show_columns(df)
			all_columns_names4 = show_columns(df)            
			selected_columns_name3 = st.selectbox("Select Columns Pearsonr",all_columns_names3)
			selected_columns_names4 = st.selectbox("Select Pearsonr Correlation",all_columns_names4)
			if st.button("Generate Pearsonr Correlation"):
				df=pd.DataFrame(Show_pearsonr(imp_df[selected_columns_name3],imp_df[selected_columns_names4]),index=['Pvalue', '0'])
				st.dataframe(df)  

			spearmanr3 = show_columns(df)
			spearmanr4 = show_columns(df)            
			spearmanr13 = st.selectbox("Select Columns spearmanr3",spearmanr4)
			spearmanr14 = st.selectbox("Select spearmanr4 Correlation",spearmanr4)
			if st.button("Generate spearmanr Correlation"):
				df=pd.DataFrame(Show_spearmanr(catego_df[spearmanr13],catego_df[spearmanr14]),index=['Pvalue', '0'])
				st.dataframe(df)  
                
			Scatter1 = show_columns(df)
			Scatter2 = show_columns(df)            
			Scatter11 = st.selectbox("Scatter1",Scatter1)
			Scatter22 = st.selectbox("Scatter2",Scatter2)
			if st.button("Generate PLOTLY Scatter PLOT"):
				st.pyplot(plotly(df,df[Scatter11],df[Scatter22]))
                
			bar1 = show_columns(df)
			bar2 = show_columns(df)            
			bar11 = st.selectbox("bar1",bar1)
			bar22 = st.selectbox("bar2",bar2)
			if st.button("Generate PLOTLY histogram PLOT"):
				st.pyplot(plotly_histogram(df,df[bar11],df[bar22]))                
                
	st.title("Credits and Inspiration")
	"""https://github.com/vishalsiram"""    


if __name__ == '__main__':
	main()