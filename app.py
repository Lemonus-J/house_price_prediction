
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns 
import pandas as pd
import numpy as np
from predict_cost import predict

st.set_page_config(layout='wide')
sns.set_theme(rc={'axes.facecolor':'#0e1117', 'figure.facecolor':'#0e1117'})

df = pd.read_excel('./dataset/HousePricePrediction.xlsx')
category = {
    'Year Remodelled' : 'YearRemodAdd',
    'Total Basement Square Feet' : 'TotalBsmtSF',
    'Overall Condition' : 'OverallCond',
    'Lot Configuration' : 'LotConfig',
    'Building Type' : 'BldgType',
    'Main Street Zoning' : 'MSZoning',
    'Exterior' : 'Exterior1st'
}
    
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ['House Price Dataset Analysis', 'House Price Prediction Simulator'])
st.sidebar.write('---')
# config
if (selection == 'House Price Dataset Analysis'):
    year_range = st.sidebar.slider('Year Range:', df['YearRemodAdd'].min(), df['YearRemodAdd'].max(), (df['YearRemodAdd'].max()-10, df['YearRemodAdd'].max()))
    st.sidebar.write("<span style='font-size: 12px;'>*If unit has been remodelled, remodelled year will be used instead.</span>", unsafe_allow_html=True)
    filtered_price = df.query('YearRemodAdd >= @year_range[0] & YearRemodAdd <= @year_range[1]')['SalePrice']
    price_count = filtered_price.value_counts()
    occurence = st.sidebar.slider('Units Sold:', price_count.min(), price_count.max(), (price_count.max()-1, price_count.max()))

def house_price_prediction_sim():
    ext = [
    'Asbestos Shingle', 'Asphalt Shingle', 'Brick Common', 'Brick Face', 
    'Concrete Block', 'Cement Board', 'Hard Board', 'Imitation Stucco', 
    'Metal Siding', 'Plywood Siding', 'Stone', 'Stucco', 
    'Vinyl Siding', 'Wood Siding', 'Wood Shingle'
    ]
    bldg = ['Single-Family Detached', 'Two-Family Conversion', 'Duplex', 'Townhouse Inside Unit', 'Townhouse End Unit']
    lot = ['Corner Lot', 'Cul-de-sac', 'Frontage on 2 Sides of Property', 'Frontage on 3 Sides of Property', 'Inside Lot']
    zoning = ['Commercial (all)', 'Floodway', 'Residential High Density', 'Residential Low Density', 'Residential Medium Density']

    st.title('House Price Prediction Simulator')

    st.write('---')

    ms_subclass = st.slider('Type of house', 10, 200, 100, 10)
    lot_area = st.slider('Lot area of house', 1000, 15000, 1500)
    overall_cond = st.slider('Condition of house', 1, 10, 5)
    year_built = st.slider('Year built', 1850, 2020, 1975)
    if (year_built < 2020):
        year_remod = st.slider('Year remodeled', year_built, 2020, year_built)
    if (year_built == 2020):
        year_remod = st.slider('Year remodeled', year_built, 2019, year_built, disabled=True)
    bsmt_fin_sf2 = st.slider('Type 2 finished square feet', 0.0, 2000.0, 0.0)
    bsmt_sf = st.slider('Basement square feet', 0.0, 6500.0, 0.0)

    ms_zoning = st.selectbox('Zone classification of sale', options=zoning)
    lot_config = st.selectbox('Lot configuration', options=lot)
    building = st.selectbox('Type of building', options=bldg)
    exterior = st.selectbox('Exterior covering on house', options=ext,)

    final = [ms_subclass, lot_area, overall_cond, year_built, year_remod, bsmt_fin_sf2, bsmt_sf] + [1 if string == ms_zoning else 0 for string in zoning] + [1 if string == lot_config else 0 for string in lot] + [1 if string == building else 0 for string in bldg] + [1 if string == exterior else 0 for string in ext]

    if st.button('Predict House Price'):
        cost = predict(np.array([final]))
        st.write('This unit can be sold for:')
        st.text(f'{round(cost[0],2)}$')
        
def house_price_dataset_analysis():
    st.title('House Price Dataset Analysis')   
    st.write('---')

    # Lot Area vs Sale Price
    with st.container():
        hue_selection = st.selectbox('Hue', options=category.keys())
        hue = category[hue_selection]
        if (hue == "LotConfig" or hue == "BldgType" or hue == "MSZoning" or hue == "Exterior1st"):
            cmap = False
        elif (hue == "YearRemodAdd" or hue == "TotalBsmtSF" or hue == "OverallCond"):
            cmap = True
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.scatterplot(
            x='SalePrice', 
            y='LotArea', 
            hue=hue, 
            data=df.query('YearRemodAdd >= @year_range[0] & YearRemodAdd <= @year_range[1]'), 
            ax=ax, 
            palette=sns.color_palette("Spectral", as_cmap=cmap)
            )

        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        ax.set_xlabel('SalePrice', color='white', fontsize=12)
        ax.set_ylabel('LotArea', color='white', fontsize=12)
        ax.set_title(f'Relationship between Lot Area and Sale Price from Year {year_range[0]}-{year_range[1]}', color='white')
        legend = plt.legend()
        for text in legend.get_texts():
            text.set_color("white")

        st.pyplot(fig)

    st.write('---')

    cols1 = st.columns(2)

    # Sale Price Distribution by year remod and count
    with cols1[0]:
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(
            x=price_count[(price_count >= occurence[0]) & (price_count <= occurence[1])].index, 
            y=price_count[(price_count >= occurence[0]) & (price_count <= occurence[1])], 
            ax=ax,
            color='#ff4b4b')
        
        ax.tick_params(axis='x', colors='white', labelsize=20, rotation=55)
        ax.tick_params(axis='y', colors='white', labelsize=20)
        ax.set_xlabel('Sale Price', color='white', fontsize=25)
        ax.set_ylabel('Count', color='white', fontsize=25)
        ax.set_title(f'Distribution of Common Sale Price From {year_range[0]}-{year_range[1]} (Sold for {occurence[0]}-{occurence[1]} Times)', color='white', fontsize=25)
        st.pyplot(fig)
    
    # Min and Max of Sale Price by year remod
    with cols1[1]:
        st.write(f'Highest and Lowest Sale Price From {year_range[0]}-{year_range[1]}')
        with st.expander(f'Highest Price: ${filtered_price.max()}'):
            max = filtered_price.max()
            df_max = df.query('SalePrice == @max')
            df_max_cleaned = df_max.drop(columns=['Id', 'SalePrice'])
            st.write(df_max_cleaned)
            
        with st.expander(f'Lowest Price: ${filtered_price.min()}'):
            min = filtered_price.min()
            df_min = df.query('SalePrice == @min')
            df_min_cleaned = df_min.drop(columns=['Id', 'SalePrice'])
            st.write(df_min_cleaned)
    
    st.write("---")
    
    cols2 = st.columns(2)

    with cols2[0]:
        fig, ax = plt.subplots(figsize=(10, 10))
        zoning_count = df.query('YearRemodAdd >= @year_range[0] & YearRemodAdd <= @year_range[1]')['MSZoning'].value_counts()
        ax.pie(zoning_count, labels=zoning_count.index, autopct='%1.1f%%', textprops={'color':'white'})
        ax.set_title(f'Distribution of Zoning Types From {year_range[0]}-{year_range[1]}', color='white', fontsize=25)
        
        st.pyplot(fig)


    with cols2[1]:
        fig, ax = plt.subplots(figsize=(10, 10))
        zoning_count = df.query('YearRemodAdd >= @year_range[0] & YearRemodAdd <= @year_range[1]')['BldgType'].value_counts()
        ax.pie(zoning_count, labels=zoning_count.index, autopct='%1.1f%%', textprops={'color':'white'})
        ax.set_title(f'Distribution of Building Types From {year_range[0]}-{year_range[1]}', color='white', fontsize=25)
        
        st.pyplot(fig)
        
    cols3 = st.columns(2)

    with cols3[0]:
        fig, ax = plt.subplots(figsize=(10, 10))
        zoning_count = df.query('YearRemodAdd >= @year_range[0] & YearRemodAdd <= @year_range[1]')['LotConfig'].value_counts()
        ax.pie(zoning_count, labels=zoning_count.index, autopct='%1.1f%%', textprops={'color':'white'})
        ax.set_title(f'Distribution of Lot Configuration From {year_range[0]}-{year_range[1]}', color='white', fontsize=25)
        
        st.pyplot(fig)


    with cols3[1]:
        fig, ax = plt.subplots(figsize=(10, 10))
        zoning_count = df.query('YearRemodAdd >= @year_range[0] & YearRemodAdd <= @year_range[1]')['Exterior1st'].value_counts()
        ax.pie(zoning_count, labels=zoning_count.index, autopct='%1.1f%%', textprops={'color':'white'})
        ax.set_title(f'Distribution of Exterior Types From {year_range[0]}-{year_range[1]}', color='white', fontsize=25)
        
        st.pyplot(fig)

if (selection == 'House Price Dataset Analysis'):
    house_price_dataset_analysis()
elif (selection == 'House Price Prediction Simulator'):
    house_price_prediction_sim()


