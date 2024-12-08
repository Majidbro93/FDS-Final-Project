
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
    


# Load the IMDB movies dataset
# We split the dataset in this project while we did analysis on the dataset

# Original datset hat was downloaded from the Kagle website
movies_df = pd.read_csv('imdb_top_1000.csv')
# Final dataset after all imutations
# I have corrected the standards fro the certificate after that and I have saved to other csv file
movies_final_def = pd.read_csv('imdb_1000_final.csv')
# dataset that with the separated genre column
movies_genre_def = pd.read_csv('imdb_1000_genre.csv')
# Corrected standrads for certfcate column based on the US 
movies_cert_mapped_def = pd.read_csv('imdb_mapped_cert_Final.csv')
# Corrected standards fro certificate column after mapping and dropping the ambiguous certificates
movies_cert_mapped_modified_def = pd.read_csv('imdb_mapped_cert_Final_modified.csv')
# Final dataset with the corrected certificate column
movies_cert_Final_with_correct_def = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

# Data cleaning for runtime and gross columns
movies_df['Runtime'] = movies_df['Runtime'].str.replace(' min', '').astype(int)
movies_df['Gross'] = movies_df['Gross'].str.replace(',', '').astype(float)

# Sidebar menu for navigation
menu = st.sidebar.selectbox(
    "Select a section",
    ["Introduction","Data Process" ,"Data Visualization", "Machine Learning For Prediction" , "Conclusion"]
)

# Introduction Section
if menu == "Introduction":
   # HTML for custom title
    import streamlit as st

# Define the clapper emoji
    clapper_emoji = "🎬"

    # HTML for custom title
    st.markdown("""
        <style>
        .custom-title {
            font-family: 'Pacifico', cursive;
            font-size: 50px;
            color: #ff6347;
        }
        </style>
        """, unsafe_allow_html=True)

    # Displaying the custom title with emoji
    st.markdown(f'<p class="custom-title">{clapper_emoji} Let\'s Talk About Movies!!</p>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    .custom-title {
        font-family: 'Pacifico', cursive;
        font-size: 30px;
        color: #ff6347;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p>By Majid Baradaran Akbarzadeh</p>', unsafe_allow_html=True)


    st.write("### Introduction")

    st.write("""
    In modern times, people have different tastes when it comes to choosing how they want to spend their free time. 
             Some prefer to spend their free time going out, reading books, or exercising. Others tend to watch series and movies a lot,
              not only because they can have a good time but also because they can learn many things from different cultures and science, or even enjoy simple comedies 
             without any special informative content, just for fun. Therefore, we can say that the film industry today plays a significant and influential role in people's lives, 
             shaping society’s culture, politics, and even economics. Furthermore, with the growing popularity of streaming platforms, the variety of genres, storytelling techniques, 
             and visual effects has made movies one of the most consumed forms of media worldwide.""")

    st.image("https://th.bing.com/th/id/R.5dd08addca8c9922408123292e2a5c3d?rik=0yiygbqJl0cAqQ&riu=http%3a%2f%2ftheseventhart.org%2fwp-content%2fuploads%2f2012%2f06%2fGodfatherIII.jpg&ehk=Em%2bD%2be0p08GrSa6RgxantpppC%2bynuuB2d7326mu3OMI%3d&risl=&pid=ImgRaw&r=0", caption="Al Pacino in a scene for GodFather III")     

    st.write("""As mentioned above, the movie industry has become incredibly important over the last few decades, and today it is a very serious business for many 
             companies and individuals around the globe. For example, you can see how the gross income of top-charting movies has increased throughout the years in the below figure. As a result, 
             many entertainment and movie corporations are investing in sophisticated plots, talented directors, and famous actors and actresses to gain more profit than their 
             competitors. Additionally, people's tastes in movie genres vary greatly, and audiences range in age from children to the elderly. In this regard, directors, producers, 
             and screenwriters must consider these factors when making movies in order to capture the audience's attention and maximize profit from their final product.""")
     

    # New Visualization 10: Total Gross Earnings by Year
    gross_by_year = movies_final_def.groupby('Released_Year')['Gross'].sum().reset_index()

    # Calculate the regression line
    slope, intercept = np.polyfit(gross_by_year['Released_Year'], np.log(gross_by_year['Gross']), 1)

    # Generate the regression line values
    regression_line = np.exp(slope * gross_by_year['Released_Year'] + intercept)

    # Create the Plotly figure with the Cividis color scale
    fig = px.line(gross_by_year, x='Released_Year', y='Gross', markers=True,
                title='Total Gross Earnings by Year',
                labels={'Released_Year': 'Release Year', 'Gross': 'Total Gross Earnings (in dollars)'},
                color_discrete_sequence=px.colors.sequential.Cividis)
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
    fig.add_scatter(x=gross_by_year['Released_Year'], y=regression_line, mode='lines', name='Regression Line',
                    line=dict(color='red', width=2))

    # Set the layout and title font sizes
    fig.update_layout(title_font_size=16)
    fig.update_yaxes(type="log")
    fig.update_xaxes(title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    fig.update_yaxes(title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    fig.update_traces(hovertemplate='Year: %{x}<br>Gross: $%{y:,}<extra></extra>')
    st.plotly_chart(fig)



    st.write("""Given the importance of the movie industry, we chose to perform data analysis on a movie dataset. There are numerous datasets available on the internet 
             for such analysis, and we selected one of the most trusted and famous sources: IMDb's [Top 1000 Movies of All Time](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows), which provides most of the information we are looking for. 
             This dataset includes various types of data, such as movie certificates, overviews, IMDb ratings, Metacritic ratings, and gross income etc. By considering these 
             variables, we can conduct an informative analysis of what makes top-rated movies stand out and provide valuable insights for both audiences and movie production companies, 
             helping them decide where to invest their time and money. This analysis is useful for both ordinary moviegoers and specialized individuals (movie critics) who follow 
             the industry closely from artistic and technical perspectives. Here, we present some initial results from the dataset to gain a better understanding of how many top-chart movies were directed by acclaimed directors 
             and which certificates are most common thoroughout the years.""")

    st.markdown("""
            Here, We have divided this project into 5 sections::
    
    - **Introduction:** We talked about the importance of the topic of this application and project.
    - **Data Processing:** We explain the statistical procedures we used to prepare our dataset for analysis, including Initial Data Analysis (IDA), 
             Exploratory Data Analysis (EDA), and data cleaning. 
    - **Visualizations:** We present interactive results from the dataset, focusing on genres, director scores, gross revenues based on genres and movie ratings and movies overview analysis.
    - **Machine Learning for predictions:** We have implemented machine learning algorithms to predict the success of movies and developed a movie recommendation system to help movie lovers discover their
     favorite films based on titles and keywords from overviews.
    - **Conclusion:** We highlight some key findings from our data analysis and the results we derived from 
             the dataset.""")
             
    st.write(""" LET'S BEGIN OUR JOURNEY INTO THE WORLD OF MOVIES! """)



    
    sns.set_style("whitegrid")
    colors = sns.color_palette("coolwarm", 10)
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    # Get the top 10 directors by number of movies
    top_directors = movies_final_def['Director'].value_counts().head(10)
    bars = ax.bar(top_directors.index, top_directors.values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_title('Top Directors with Most Movies in Top 1000', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Director', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Movies', fontsize=14, fontweight='bold', labelpad=10)

    # Rotate x-axis labels for better readability and increase font size
    ax.set_xticklabels(top_directors.index, rotation=45, ha='right', fontsize=12, fontweight='bold')

    # Add gridlines for better visualization, making them subtle and clean
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Add value labels on top of the bars for clarity
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval), ha='center', 
                fontsize=12, fontweight='bold', color='black')
    st.pyplot(fig)


    # Interactive Pie Chart (Movie Certificates)
    # Count the number of occurrences of each certificate
    certificate_counts = movies_cert_Final_with_correct_def['Certificate'].value_counts().reset_index()
    certificate_counts.columns = ['Certificate', 'Count']
    # Create the 3D-like pie chart
    fig = px.pie(certificate_counts, values='Count', names='Certificate',
                title='Distribution of Movie Certificates',
                hover_data=['Count'], hole=0.4)
    fig.update_traces(textinfo='percent+label', pull=[0.1] * len(certificate_counts), rotation=45)
    st.plotly_chart(fig)


###################################################################### Model Performance Section ############################################################################
elif menu == "Data Process":
        
        
    st.title("Data Process")

    st.write("""In this section, we will discuss how we conducted the Initial Data Analysis (IDA) and Exploratory Data Analysis (EDA) for our dataset in this project.
                Additionally, we will explain the steps we took for data cleaning and handling missing data in the dataset.""")

    # Display the first few rows of the dataset
    st.write("### Variables in the datset and type of the data")
    st.write("""We used the Top 1000 Movies of IMDb dataset for this project. In this regard, we introduce the variables we are working with in the project. 
                Below are the first few rows and columns of the dataset.""")
    st.write(movies_df.head())
    

    # Display the text with Streamlit
    st.markdown("""
        As you can see from the table above, we are working with **16 variables** in this dataset. These variables include:

        - **Poster_Link**: A link to the movie's poster
        - **Series_Title**: The movie title
        - **Released_Year**: The release year of the movie
        - **Certificate**: Indicates the authorized age range for viewing the movie
        - **Runtime**: The duration of the movie
        - **Genre**: The genres of the movie (some movies have multiple genres)
        - **IMDB_Rating**: The IMDb score of the movie
        - **Overview**: A brief summary of the movie
        - **Meta_score**: The Metacritic website score for each movie
        - **Star1 to Star4**: The actors and actresses who appeared in the movie
        - **No_of_Votes**: The number of people who voted for the movie
        - **Gross**: The gross revenue of the movie

        Now, we will introduce the types of data we are working with in this project, as shown in the table below.
        """)


    # Type of the data in the dataset
    df = pd.read_csv('imdb_1000_final.csv') 
    st.write("Data types of the columns in the dataset:")
    st.write(df.dtypes)
    st.write("""As you can see, out of the 16 different variables we are working with in this project, only six are numerical, and two of these numerical variables—released year and runtime—are categorical in nature. 
                Additionally, both certificate and genres are also categorical variables.""")
        
    st.write("### Heatmap of Missing Values")
    
    st.write("""Next, the heatmap for the missing values in the original dataset (before imputation and cleaning) is presented below. From the heatmap, 
                we can see that the missing values are in the Certificate (categorical data), Meta_score (numerical data), and Gross (numerical data) columns. 
                In the remainder of this section, we will explain how we handled the missing values and the 
                different procedures we followed to prepare the dataset for imputation using various methods.""")    

        # Visualization 1: Heatmap of Missing Values
    st.write("""Below figure shows the missing values in the original dataset (top 1000 IMDB movies). Based on the heatmap figure for missing values, we see that we have the missing 
        data.""")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(movies_df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title('Missing Values Heatmap For Original Dataset')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### Correlation Matrix For Original Dataset")
    st.write("""The figure below shows the correlation matrix for the numerical variables in the dataset we are working on. 
                The purpose of this is to gain an initial understanding of how the numerical variables are related to each other. 
                After imputation, we aim to preserve these relationships as much as possible, 
                ensuring that the correlations between the variables are not drastically altered.""")
        
        # Visualization 2: Correlation Matrix
    numerical_columns = ['IMDB_Rating', 'Runtime', 'Gross', 'Meta_score', 'No_of_Votes']
    corr_matrix = movies_df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

    st.write("### Data Cleaning ")
    st.write("""In our dataset, we do not have any duplicate numerical data. However, in the Certificate column, which indicates the movie's viewing allowance for specific audiences (age range), we have ratings listed with different standards, each with its own sign and convention. Another issue arises in the Genre column, where some movies are assigned multiple genres, separated by commas. This creates a challenge for the imputation process, as we need to separate these genres into different columns. To address
                both issues, we first used mapping to standardize the sign conventions in the Certificate column and then split the Genre column 
                into separate columns for each genre.""")

        # Creating a count plot to visualize the relationship between 'Genre_1' and 'Certificate'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=movies_genre_def, x='Genre_1', hue='Certificate')
    plt.title("Distribution of 'Main Genre' across different 'Certificate' categories before mapping")
    plt.xlabel("Primary Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    st.write("""As you see from the figure above, we have different sign and convention for movies ratings based on different age ranges. In this regard, we considered the movie ratings 
        that used in the United States which we can see them on the IMDB website in the sction of movie ratings. """)
    st.markdown("""
        These ratings are as follows:

        - **G**: For all audience
        - **PG**: Parental Guidance Suggested (mainly for under 10's)
        - **PG-13**: Parental Guidance Suggested for children under 13
        - **R**: Under 17 not admitted without parent or guardian
        - **Approved**: Pre-1968 titles only (from the MPA site) Under the Hays Code, films were simply approved or disapproved based on whether they were deemed 'moral' or 'immoral'.)
        """)
    st.write("""So, we mapped all the other ratings in the dataset based on the same or similar categories used in the United States. Moreover, some ratings, 
                such as Passed, UA, and U/A, were considered ambiguous in meaning and definition. Therefore, we treated them as NaN and will attempt to impute them using 
                the classifier imputer technique called Random Forest. Below, you can see the bar plot showing the count of different certificates
                based on the main genre of the film after mapping.""")
        
        # Creating a count plot to visualize the relationship between 'Genre_1' and 'Certificate'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=movies_cert_mapped_def, x='Genre_1', hue='Certificate')
    plt.title("Distribution of 'Genre_1' across different 'Certificate' categories after mapping")
    plt.xlabel("Primary Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Splitting the 'Genre' column into multiple columns based on ','
    genre_split = movies_df['Genre'].str.split(',', expand=True)
    genre_split.columns = [f'Genre_{i+1}' for i in range(genre_split.shape[1])]
    imdb_1000_gensp_df = pd.concat([movies_df, genre_split], axis=1)
        # Plot heatmap to visualize missing values
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(movies_cert_mapped_modified_def.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Heatmap of Missing Values of the gensep dataset')
    plt.xticks(rotation=45)
    st.pyplot(fig)

        # Load your IMDb dataset
    df = movies_cert_mapped_modified_def

        # Count missing values in each column
    missing_values = df.isnull().sum()

        # Display the missing values count in Streamlit
    st.write("### Missing Values Count")
    st.write(missing_values)



    st.write("### Handling missing values")
    st.write("""We have missing values in three columns, as shown in the heatmap for missing data. Specifically, there are missing values 
                in the Certificate, Meta_score, and Gross columns of the dataset. To address this issue, we have considered several methods for handling the missing values in these columns.
                First, we will explain how we perform imputation for the numerical columns, such as Meta_score and Gross.""")
    st.write("### Handling missingness in Meta score column")
    st.write("""In this section we will discuss that how we impute the missing values in the Meta_score column by using two diffrent methods. These two approaches are
                Median and K-nearest neighbor (KNN). """)
    st.write("""In the plots below, we see two different comparisons of the dataset before and after imputation using the Median and KNN methods. Our KNN imputation was based on the 
                IMDB_Rating, No_of_Votes, and Meta_score columns. Since IMDB_Rating and Meta_score are closely related for the top 1000 movies on the IMDb chart, we applied the median method 
                based on the IMDB_Rating column. From the histograms, we observe a better spread with the KNN method compared to the median method, where the peak values are higher. Additionally,
                in the scatter plots, 
                we again see a better spread with the KNN method compared to the median method, where the latter shows straight lines of data points among the original dataset's data points.
                """)
        
    ### Using KNN for findinf the missingness in the Meta_score column
   
        

    # Selecting relevant numerical features
    features = ['IMDB_Rating', 'No_of_Votes', 'Meta_score']  # Features including 'Meta_score'
    df_knn = imdb_1000_gensp_df[features].copy()

        # Applying KNN Imputation
    imputer = KNNImputer(n_neighbors=10)
    df_knn_imputed = pd.DataFrame(imputer.fit_transform(df_knn), columns=features)


    imdb_1000_KNNMeta_df = imdb_1000_gensp_df.copy()

    imdb_1000_KNNMeta_df['Meta_score'] = df_knn_imputed['Meta_score']
        
        ## Using Median for finding missingness in Meta_score column


    imdb_1000_MedMeta_df = imdb_1000_gensp_df.copy()
        # Fill missing Meta_score values with the median Meta_score within each genre in the copied dataset
    imdb_1000_MedMeta_df['Meta_score'] = imdb_1000_MedMeta_df.groupby('IMDB_Rating')['Meta_score'].transform(lambda x: x.fillna(x.median()))

        ## Plots for comparing KNN and median imoutation for Meta_score column

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for the KNN imputed Meta_score on the first subplot
    imdb_1000_gensp_df['Meta_score'].plot(kind='hist', alpha=0.5, color='blue', label='Original (Missing Values)', bins=30, ax=axes[0])
    imdb_1000_KNNMeta_df['Meta_score'].plot(kind='hist', alpha=0.5, color='green', label='Imputed (KNN)', bins=30, ax=axes[0])
    axes[0].set_title('Histogram Comparison of Meta_score (KNN Imputation)')
    axes[0].set_xlabel('Meta_score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

        # Histogram Plot for the Median imputed Meta_score on the second subplot
    imdb_1000_gensp_df['Meta_score'].plot(kind='hist', alpha=0.5, color='blue', label='Original (Missing Values)', bins=30, ax=axes[1])
    imdb_1000_MedMeta_df['Meta_score'].plot(kind='hist', alpha=0.5, color='green', label='Imputed (Median)', bins=30, ax=axes[1])
    axes[1].set_title('Histogram Comparison of Meta_score (Median Imputation)')
    axes[1].set_xlabel('Meta_score')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()


    plt.tight_layout()
    st.pyplot(fig)

        # Scattered plots for comparing the two methods for the imoutation on Meta_core
        

    imputed_mask = imdb_1000_gensp_df['Meta_score'].isnull()


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot for KNN imputed Meta_score on the first subplot
    axes[0].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_KNNMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_KNNMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Meta_score (KNN)')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Meta_score')
    axes[0].legend()

        # Scatter plot for Median imputed Meta_score on the second subplot
    axes[1].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_MedMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_MedMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Meta_score (Median)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Meta_score')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)



    st.write("""In the scatter plots below, we compare the imputed Meta_score data points using the KNN and median methods. We observe a 
                better trend and distribution of the imputed data points with the KNN method compared to the median method. Although the error bounds for 
                the KNN and median methods are similar, at 15.35% and 
                13.93%, respectively, the distribution trend of the KNN method is better than that of the median method.""")


    # Scatter plot median Vs KNN for Meta_score in terms of No_of_votes

        # Create a mask to identify the imputed values (previously missing values)
    imputed_mask = imdb_1000_gensp_df['Meta_score'].isnull()

        # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot for KNN imputed Meta_score on the first subplot
    axes[0].scatter(imdb_1000_KNNMeta_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_KNNMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_KNNMeta_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_KNNMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Meta_score (KNN)')
    axes[0].set_xlabel('No of Votes')
    axes[0].set_ylabel('Meta_score')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True)

        # Scatter plot for Median imputed Meta_score on the second subplot
    axes[1].scatter(imdb_1000_MedMeta_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_MedMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_MedMeta_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_MedMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Meta_score (Median)')
    axes[1].set_xlabel('No of Votes')
    axes[1].set_ylabel('Meta_score')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True)

        
    plt.tight_layout()

    st.pyplot(fig)


    ###### Gross imputation
    st.write("### Handling missingness in Gross column")
    st.write("""In this section, we applied the same procedure to the Gross income column using two different methods: the Median and K-nearest neighbor (KNN) approaches. """)
    st.write("""Based on the plots below, we observe a better trend for imputing the missing data using KNN for the Gross income of the movies, 
                whereas with the median method, only one point was imputed. Additionally, from the scatter plot using KNN for imputation, we see that the 
                imputed data points follow the same trend as the general dataset.
                Therefore, for Gross income, we chose KNN as the better imputation method.""")
    ## KNN method
        # Selecting relevant numerical features
    features = ['IMDB_Rating', 'No_of_Votes', 'Gross']  # Features including 'Meta_score'
    df_knn = imdb_1000_gensp_df[features].copy()

        # Applying KNN Imputation
    imputer = KNNImputer(n_neighbors=10)
    df_knn_imputed = pd.DataFrame(imputer.fit_transform(df_knn), columns=features)


    imdb_1000_KNNGross_df = imdb_1000_gensp_df.copy()

    imdb_1000_KNNGross_df['Gross'] = df_knn_imputed['Gross']

        ## Median method

    imdb_1000_MedGross_df = imdb_1000_gensp_df.copy()
        # Fill missing Meta_score values with the median Meta_score within each genre in the copied dataset
    imdb_1000_MedGross_df['Gross'] = imdb_1000_MedGross_df.groupby('No_of_Votes')['Gross'].transform(lambda x: x.fillna(x.median()))


        # Create a mask to identify the imputed values (previously missing values)
    imputed_mask = imdb_1000_gensp_df['Gross'].isnull()

        # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot for KNN imputed Gross on the first subplot
    axes[0].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_KNNGross_df.loc[~imputed_mask, 'Gross'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_KNNGross_df.loc[imputed_mask, 'Gross'], 
                        color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Gross (KNN)')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Gross')
    axes[0].set_yscale('log')
    axes[0].legend()

    # Scatter plot for Median imputed Gross on the second subplot
    axes[1].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_MedGross_df.loc[~imputed_mask, 'Gross'], 
                    color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_MedGross_df.loc[imputed_mask, 'Gross'], 
                    color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Gross (Median)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Gross')
    axes[1].set_yscale('log')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)


    imputed_mask = imdb_1000_gensp_df['Gross'].isnull()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot for KNN imputed Gross on the first subplot
    axes[0].scatter(imdb_1000_KNNGross_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_KNNGross_df.loc[~imputed_mask, 'Gross'], 
                    color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_KNNGross_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_KNNGross_df.loc[imputed_mask, 'Gross'], 
                    color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Gross (KNN)')
    axes[0].set_xlabel('No of Votes')
    axes[0].set_ylabel('Gross')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True)

    # Scatter plot for Median imputed Gross on the second subplot
    axes[1].scatter(imdb_1000_MedGross_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_MedGross_df.loc[~imputed_mask, 'Gross'], 
                    color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_MedGross_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_MedGross_df.loc[imputed_mask, 'Gross'], 
                    color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Gross (Median)')
    axes[1].set_xlabel('No of Votes')
    axes[1].set_ylabel('Gross')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

###### Certificate imputation
    st.write("### Handling missingness in Certificate column")
    st.write("""In this section, we will explain the process we followed to identify the missing values in the Certificate column. Since the Certificate column is not numeric, 
            we needed to use a method that handles missing values for categorical data. Here, we used the Random Forest Classifier (RFC) to impute the missing values in this column. 
            For this, we encoded variables such as Genre_1 (main genre of the movie), Director, Star1 (leading actor), Star2 (supporting actor 1), Star3 (supporting actor 2), and Star4 
            (supporting actor 3). Using the RFC method,
            we divided the dataset into training and test data points and allowed the model to learn how to predict the correct Certificate based on the mentioned variables.""")


    ## Comparing the correlation matrices

    

    # Ensure 'Gross' and 'Runtime' are strings before replacing characters for the original dataset
    movies_df['Gross'] = movies_df['Gross'].astype(str).str.replace(',', '').astype(float)
    movies_df['Runtime'] = movies_df['Runtime'].astype(str).str.replace(' min', '').astype(float)

    # Ensure 'Gross' and 'Runtime' are strings before replacing characters for the imputed dataset
    movies_final_def['Gross'] = movies_final_def['Gross'].astype(str).str.replace(',', '').astype(float)
    movies_final_def['Runtime'] = movies_final_def['Runtime'].astype(str).str.replace(' min', '').astype(float)

    # Select relevant columns for correlation
    selected_columns = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']

    correlation_matrix_org = movies_df[selected_columns].corr()

    correlation_matrix_gensp = movies_final_def[selected_columns].corr()

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the correlation matrix for original dataset
    sns.heatmap(correlation_matrix_org, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Correlation Matrix: Original Dataset")

    # Plot the correlation matrix for imputed dataset
    sns.heatmap(correlation_matrix_gensp, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Correlation Matrix: Imputed Dataset")

    plt.tight_layout()
    st.pyplot(fig)




    # Plot heatmap for the final dataset
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(movies_final_def.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Heatmap of Missing Values of the gensep dataset')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### Principle Component Analysis")
    st.write("""In this section, we want to apply more sophisticated analysis on our dataset to get more information that how each component in our datset can contribute to the final
             results and conclusions. Principle Component Analysis (PCA) is one of the famous techniques for dimensionality reduction that
              transforms our dataset into a set of linearly uncorrelated variables (principal components) while retaining most of the variance in the data. In this regard, by using this method we can
            patterns and relationships in our dataset to improve computational efficiency.""")
    
    st.write("""Based on the plot below, we see the screeplot that shows the explained variance ratio for our dataset which each point shows the priciple component (PC) that are the 
             5 numrical features in our dataset. In this regard, we see that the PC1 and PC2 are capturing more than 0.6 of the explained variance in our dataset and also by adding
             another PC (PC3) we see that it grows to 0.8. Based on this plot, one can conclude that we can use the reduction in our dimensionality for the analysis that we will do.  """)
    

    
    

    # Load the dataset
    data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

    # Select numeric features for PCA
    features = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross', 'Runtime']
    numeric_data = data[features].dropna()  # Drop rows with missing values

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Apply PCA
    pca = PCA(n_components=len(features))  # Keep all components
    pca_result = pca.fit_transform(scaled_data)

    # Screeplot: Variance explained by each principal component
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Streamlit layout
    #st.title('Screeplot: Explained Variance by Principal Components')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot for individual explained variance
    ax.scatter(range(1, len(features) + 1), explained_variance, color='blue', label='Individual Explained Variance', s=100)
    ax.plot(range(1, len(features) + 1), explained_variance, color='blue', linestyle='--')

    # Scatter plot for cumulative explained variance
    ax.scatter(range(1, len(features) + 1), cumulative_variance, color='red', label='Cumulative Explained Variance', s=100)
    ax.plot(range(1, len(features) + 1), cumulative_variance, color='red', linestyle='--')

    ax.set_xticks(range(1, len(features) + 1))
    ax.set_xticklabels([f'PC{i}' for i in range(1, len(features) + 1)])
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xlabel('Principal Components')
    ax.legend(loc='best')
    ax.grid()

    # Display the plot in Streamlit
    st.pyplot(fig)


    st.write("""Now we see the Biplot based on first two principle components for our dataset. In this plot, the arrows show each numerical variable in our dataset and how they are
             correlated with each other. The closer the angle of those arrows the more correlated the corresponding features of the dataset. In this regard, each two arrows that are closer 
             such as IMDB_Rating and Meta_score are positively correlated.""")




    # Load the dataset
    data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

    # Select numeric features for PCA
    features = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross', 'Runtime']
    numeric_data = data[features].dropna()  # Drop rows with missing values

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Apply PCA
    pca = PCA(n_components=len(features))  # Keep all components
    pca_result = pca.fit_transform(scaled_data)

    # Biplot: First two principal components
    def biplot(scores, coeffs, feature_names, pc_x=0, pc_y=1):
        """
        Create a Biplot with the first two principal components.

        Args:
        - scores: The PCA-transformed data.
        - coeffs: PCA component coefficients (loadings).
        - feature_names: List of feature names.
        - pc_x, pc_y: Indices of the principal components to plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot of scores
        ax.scatter(scores[:, pc_x], scores[:, pc_y], alpha=0.6, edgecolors='k', label='Data points')

        # Feature vectors
        for i, feature in enumerate(feature_names):
            ax.arrow(0, 0, coeffs[i, pc_x]*2, coeffs[i, pc_y]*2, color='red', alpha=0.8, head_width=0.05)
            ax.text(coeffs[i, pc_x]*2.2, coeffs[i, pc_y]*2.2, feature, color='black', ha='center', va='center')

        ax.set_xlabel(f'Principal Component {pc_x + 1}')
        ax.set_ylabel(f'Principal Component {pc_y + 1}')
        ax.grid()
        ax.legend()

        return fig

    # Create the biplot
    fig = biplot(pca_result, pca.components_.T, features)

    # Display the plot in Streamlit
    #st.title('Biplot: First Two Principal Components')
    st.pyplot(fig)

###########################################################

    

    st.write("### IDA and EDA for the overview of the movies:")

    st.write("#### Movie Sentiment Analysis:")
    st.write("""In the graph below, you can see histograms combined with Kernel Density Estimation (KDE) plots for both Sentiment Polarity and Sentiment Subjectivity distributions. 
    The x-axis can be adjusted to switch between these two metrics. For the Sentiment Polarity Distribution, the graph shows the spread of sentiment polarity values across all movie overviews, providing insight into their tonal characteristics.""") 
    
    st.markdown("""
    In this regard, the polarity ranges as below:
    
    - **Negative polarity (<0):** Indicates negative sentiments in overviews.
    - **Neutral polarity (≈0):** Indicates neutral or balanced overviews. 
    - **Positive polarity (>0):** Indicates positive sentiments in overviews.""")

    st.write("""As we change the x-axis, we see that the histogram showing distribution of subjectivity values for all movie overviews.""") 
    
    st.markdown("""
    In this regard, the subjectivity value ranges are as below:
    
    - **Low subjectivity (<0.5):** Overviews are more objective (fact-based).
    - **High subjectivity (>0.5):** Overviews are more subjective (opinion-based).""")

    st.write("""From the sentiment polarity graph, we observe that the majority of movie overviews exhibit neutral polarity, forming a normal distribution. One reason for this is 
    that overviews are typically written by the movie's creators or marketing teams, who avoid revealing too many details of the story and therefore use more neutral language.
    Similarly, from the sentiment subjectivity graph, we see that most movie overviews have a subjectivity score below 0.5, indicating a tendency toward fact-based descriptions. 
    This is likely because these overviews are intended to provide an initial idea of the movie to potential viewers, prompting the use of objective, fact-based language to inform rather 
    than influence.""")




    # Load the dataset
    @st.cache_data
    def load_data():
        data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')
        data['Overview'] = data['Overview'].fillna('')  # Ensure no missing values in the Overview column
        return data

    data = load_data()

    # Step 2: Sentiment Analysis Function
    @st.cache_data
    def analyze_sentiment(data):
        def sentiment_analysis(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity

        data['Sentiment_Polarity'], data['Sentiment_Subjectivity'] = zip(*data['Overview'].apply(sentiment_analysis))
        return data

    data = analyze_sentiment(data)

    # Streamlit App Layout
    #st.title("Movie Sentiment Analysis")
    #st.write("Analyze the sentiment of movie overviews with interactive visualizations.")

    # User input for x-axis selection
    #st.subheader("Select Sentiment Metric for Distribution Plot")
    x_axis = st.radio(
        "Choose the sentiment metric to plot:",
        options=['Sentiment_Polarity', 'Sentiment_Subjectivity'],
        index=0
    )

    # Plot histogram with KDE using Seaborn
    st.subheader(f"Distribution of {x_axis.replace("_", " ")} with KDE Overlay")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data[x_axis],
        kde=True,  # Add KDE curve
        bins=30,
        color='blue',
        ax=ax
    )
    ax.set_title(f'Distribution of {x_axis} with KDE Overlay', fontsize=16)
    ax.set_xlabel(x_axis, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig)
    ###########################


    # Load the dataset
    @st.cache_data
    def load_data():
        data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')
        data['Overview'] = data['Overview'].fillna('')  # Ensure no missing values in the Overview column
        return data

    data = load_data()
    st.write("#### Movie Sentiment Analysis:")
    st.write("""In the below graph, we see that the most common bigrams (two words together) in the overview of the top ranked movies. One can see they are very wide vriety of movies 
    with different topics appeared in here. So, we can get the general idea of what are main topics in the overviews of the movies. """)
    # Streamlit App Layout
    #st.title("N-Gram Analysis (Bigrams) in Movie Overviews")
    #st.write("Analyze the most common bigrams (two-word combinations) in the movie overviews.")

    # Step 6: N-Gram Analysis (Bigrams)
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    X = vectorizer.fit_transform(data['Overview'])

    # Convert the sparse matrix to a dense array and sum along columns
    bigrams_sum = np.asarray(X.sum(axis=0)).flatten()

    # Create a DataFrame with the bigrams and their frequencies
    bigrams_df = pd.DataFrame({'Bigram': vectorizer.get_feature_names_out(), 'Frequency': bigrams_sum})
    bigrams_df = bigrams_df.sort_values(by='Frequency', ascending=False).head(20)

    # Display the DataFrame in Streamlit
    st.subheader("Top 20 Most Common Bigrams")
    st.dataframe(bigrams_df)

    # Plot the most common bigrams
    #st.subheader("Visualization of Most Common Bigrams")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Bigram', data=bigrams_df, palette='viridis', ax=ax)
    ax.set_title("Most Common Bigrams in Overviews")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Bigram")
    st.pyplot(fig)

    


    













###########################################################################################################################################################
############################################################## Data Visualization Section #################################################################
###########################################################################################################################################################
elif menu == "Data Visualization":
    
    st.title("Visualizations of the IMDB Top 1000 Movies")

    st.write("""In this section, we will provide you with some informative and interesting results through interactive visualizations. Our analysis was based on the variables in the top 1000 movies dataset from IMDb. In this regard, our focus was on how different genres of movies evolved and how they can affect the movie market in terms of gross revenue and scores from IMDb and Metacritic. Moreover, we have analyzed the performance of different directors and actors, examining how they influence movie performance based on the same characteristics, such as movie scores and income. We should mention that we designed this application to cater to a wide audience, ranging from ordinary people who want to gain some knowledge about the movie industry to those in charge of movie corporations,
              helping them decide which movie genre, certificate, director, and actor are worth investing in to gain the most profit and make their final
              product stand out from the crowd.""")


    st.write("### Top Genres Over the Decades and their popularity")
    
    st.write("""The graph below shows a histogram plot for different genres to visualize the rate of change in the number of movies produced with a specific genre over different decades. You can select the genre you want from the drop-down menu and see how many movies of that specific genre entered the market and charted in the top 1000. For example, with the advent of new technology in the movie industry, we observe an increase in the number of Sci-Fi movies produced from 1980 to 2020. This indicates that the movie industry decided to produce more Sci-Fi movies as they became more feasible. Additionally, 
             you can compare the gross revenue of each genre over the decades simultaneously. This allows you to observe the correlation between movie genres and their
              gross income over the years.  """)


    # Load dataset
    df = pd.read_csv('imdb_1000_final.csv')  # Ensure your file path is correct
    # Split the 'Genre' column
    df_genres = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')
    # Convert 'Released_Year' and 'Gross' to numeric
    df_genres['Released_Year'] = pd.to_numeric(df_genres['Released_Year'], errors='coerce')
    df_genres['Gross'] = pd.to_numeric(df_genres['Gross'], errors='coerce')
    # Create a new column to group movies into decades
    df_genres['Decade'] = (df_genres['Released_Year'] // 10) * 10
    available_genres = df_genres['Genre'].unique().tolist()
    st.write("Select genres to compare:")
    genres_to_compare = st.multiselect("Choose genres", available_genres, default=[available_genres[0],available_genres[6]])

    # Visualization 1: Genres over decades (Histogram of released years)
    if genres_to_compare:
        df_selected_genres = df_genres[df_genres['Genre'].isin(genres_to_compare)]
        decade_bins = np.arange(1920, 2030, 10)  
        plt.figure(figsize=(12, 7))

        sns.histplot(data=df_selected_genres, x='Released_Year', hue='Genre', hue_order=genres_to_compare,
                    bins=decade_bins, multiple='dodge', shrink=0.8, palette='Set2', edgecolor='black', linewidth=1.2)

        plt.xlabel('Released Year (by Decade)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Movies', fontsize=14, fontweight='bold')
        plt.title('Comparison of Released Years (by Decade) for Selected Genres', fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(title='Genre', fontsize=12, title_fontsize=14)
        st.pyplot(plt.gcf())

        # Visualization 2: Genres popularity ("Decades vs. Gross Income for Selected Genres")
        plt.figure(figsize=(12, 7))
        gross_by_decade = df_selected_genres.groupby(['Decade', 'Genre'])['Gross'].sum().reset_index()
        sns.lineplot(data=gross_by_decade, x='Decade', y='Gross', hue='Genre', marker='o', palette='Set2')
        plt.xlabel('Decade', fontsize=14, fontweight='bold')
        plt.ylabel('Total Gross Income', fontsize=14, fontweight='bold')
        plt.title('Decades vs. Gross Income for Selected Genres', fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(title='Genre', fontsize=12, title_fontsize=14)
        st.pyplot(plt.gcf())
    else:
        st.write("Please select at least one genre from the dropdown above.")


#################################################
#################################################
##################################################

    st.write("### Gross Income and Number of Votes by Certificate Over Decades")
    st.write("""The graph below shows the kernel density estimate (KDE) plot, illustrating how the data is distributed for different movie certificates 
             in terms of either gross revenue or the number of votes by people on IMDb. The peak values for each movie certificate indicate that a specific certificate gained popularity 
             during a particular decade based on the above-mentioned variables. Moreover, you may observe shifts in specific certificates over the range of decades, revealing how the 
             popularity of certain movie certificates increased or decreased over time. Additionally, the width of the plots indicates
              how many movies with a specific certificate were produced. Furthermore, you can change the y-axis to switch between gross revenue and the number of votes from the audience.""")



# Visualization 3: Normalized Gross Revenue and Normalized Number of Votes by Certificate Over Decades

    # Load your IMDb dataset
    df = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

    # Convert 'Released_Year', 'Gross', and 'No_of_Votes' to numeric
    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
    df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')

    # Create a new column to group movies into decades
    df['Decade'] = (df['Released_Year'] // 10) * 10

    # Normalize 'Gross' and 'No_of_Votes'
    df['Normalized Gross Revenue'] = df['Gross'] / df['Gross'].sum()
    df['Normalized Number of Votes'] = df['No_of_Votes'] / df['No_of_Votes'].sum()

    available_certificates = df['Certificate'].unique().tolist()

    # Allow certificate selection
    st.write("Select certificates to compare:")
    certificates_to_compare = st.multiselect("Choose certificates", available_certificates, 
                                            default=[available_certificates[0], available_certificates[2], available_certificates[4]], key='ghgh11111')
    y_axis_option = st.radio("Select the metric to plot on the Y-axis:", ('Normalized Gross Revenue', 'Normalized Number of Votes'))
    
    if certificates_to_compare:
        df_selected_certificates = df[df['Certificate'].isin(certificates_to_compare)]


        palette = sns.color_palette("husl", len(certificates_to_compare))

        # Plot the KDE curve with shading
        plt.figure(figsize=(10, 6))

        for idx, certificate in enumerate(certificates_to_compare):
            df_certificate = df_selected_certificates[df_selected_certificates['Certificate'] == certificate]

            total_weight = df_certificate[y_axis_option].sum()
            normalized_weights = df_certificate[y_axis_option] / total_weight if total_weight > 0 else df_certificate[y_axis_option]

            # Plot the KDE curve with normalized weights
            sns.kdeplot(data=df_certificate, x='Decade', weights=normalized_weights, fill=True, 
                        color=palette[idx], label=f'{certificate}', alpha=0.6)
        plt.xlabel('Decade', fontweight='bold')  
        plt.ylabel(y_axis_option, fontweight='bold')  
        plt.title(f'Distribution of {y_axis_option} over Decades for Selected Certificates', fontweight='bold', fontsize=14)  # Bold title
        plt.legend(title='Certificate', title_fontsize='13', fontsize='11', loc='upper right')  # Bold legend title and larger font

        plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3) 
        st.pyplot(plt.gcf())
    else:
        st.write("Please select at least one certificate from the menu above.")



###############################################################################################
# Visualization 4: comparing directors IMDB scores and gross income for different movies

    st.write("### Top directors performance over time based on gross revenue and IMDB ratings")
    st.write("""This is one of the most interesting graphs in the application. We can see how the performance of top directors changed over time using a dual axis, 
             displaying both the gross revenue of their movies and IMDb scores. If you hover over each point on the graph, you can see additional information, such as 
             the name of the movie directed by that specific director. Here, we observe that movies with higher IMDb scores tend to generate less revenue compared to the 
             director's other movies or even compared to other directors' films. In our view, this may be due to several reasons, such as movies with higher IMDb scores gaining 
             attention after their theatrical release or becoming popular years later from the audience's perspective. Conversely, we notice the opposite trend for movies with lower 
             scores but higher gross revenue. Moreover, some directors performed better with certain movies, while others failed to capture the audience's attention 
             with their other films, which explains the trends seen in this visualization.""")

    import plotly.graph_objects as go

    df = pd.read_csv('imdb_1000_final.csv')

    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')

    # Sidebar for director selection
    directors = df['Director'].unique().tolist()
    st.write("Select Directors for Comparing IMDb Ratings and Gross Income")
    selected_directors = st.multiselect("Select directors", directors, default=[directors[1],directors[6]])

    # Filter the dataset based on the selected directors
    df_directors = df[df['Director'].isin(selected_directors)]

    # Sort the movies by release year
    df_directors = df_directors.sort_values(by='Released_Year')

    # Plot the IMDb rating and gross income trend over time for the selected directors
    if not df_directors.empty:
        # Create the figure using Plotly Graph Objects
        fig = go.Figure()

        for director in selected_directors:
            df_director = df_directors[df_directors['Director'] == director]

            fig.add_trace(
                go.Scatter(x=df_director['Released_Year'], y=df_director['IMDB_Rating'], 
                        mode='lines+markers', name=f'IMDb Rating: {director}', line=dict(shape='linear'),
                        hovertemplate='<b>%{text}</b><br>IMDb Rating: %{y}<br>Year: %{x}',
                        text=df_director['Series_Title'])
            )

            fig.add_trace(
                go.Scatter(x=df_director['Released_Year'], y=df_director['Gross'], 
                        mode='lines+markers', name=f'Gross Income: {director}', line=dict(shape='linear', dash='dash'),
                        yaxis='y2',  # This specifies the second y-axis
                        hovertemplate='<b>%{text}</b><br>Gross: $%{y}<br>Year: %{x}',
                        text=df_director['Series_Title'])
            )

        fig.update_layout(
            title="IMDb Ratings and Gross Income over Time for Selected Directors",
            xaxis_title="Released Year",
            yaxis_title="IMDb Rating",
            yaxis2=dict(title="Gross Income ($)", overlaying='y', side='right'),
            legend_title="Metrics",
            hovermode="x unified"
        )
        st.plotly_chart(fig)

    else:
        st.write("Please select at least one director.")

    

    
    
    
# Visualization 5: 


    st.write("### Top IMDb Rating Movies by Star")
    st.write("""For this part of the visualization, we aim to provide an overview of each actor or actress in this dataset. By selecting your favorite movie star, 
             you can view their movie title(s), gross revenue, release date, IMDb rating, and the movie’s rank in the top IMDb chart. This gives a brief resume of that specific 
             movie star and the quality of the movies they have appeared in. Moreover, since this application is open to the public, producers and movie corporations can use 
             this part of the application for a quick review of the person they are considering for a contract. In this regard, it provides convenient access to the background of the movie star.""")


###########################################
    

    # Star selection
    star = st.selectbox('🌟 Select your favorite celebrity', pd.concat([movies_final_def['Star1'], movies_final_def['Star2'], movies_final_def['Star3'], movies_final_def['Star4']]).unique(), key="u2")

    # Filter the dataset for movies where the actor appeared
    filtered_df = movies_final_def[
        (movies_final_def['Star1'] == star) | 
        (movies_final_def['Star2'] == star) | 
        (movies_final_def['Star3'] == star) | 
        (movies_final_def['Star4'] == star)
    ].sort_values('IMDB_Rating', ascending=False)

    # Display top movies by IMDb rating
    st.markdown(f"## **Top Movies for _{star}_ based on IMDb Rating**")

    styled_df = filtered_df[['Series_Title', 'IMDB_Rating', 'Released_Year', 'Gross']].style \
        .format({
            'IMDB_Rating': '{:.1f}', 
            'Gross': '${:,.0f}'  
        }) \
        .set_properties(**{
            'background-color': '#F5F5F5',  
            'color': 'black',  
            'border-color': '#7F7F7F',  
            'font-size': '15px',  
            'font-family': 'Verdana',  
            'border': '2px solid #7F7F7F'  
        }) \
        .highlight_max(subset='IMDB_Rating', color='lightgreen') \
        .highlight_min(subset='IMDB_Rating', color='lightcoral') \
        .bar(subset='Gross', color='skyblue', vmin=0) \
        .set_table_styles([{
            'selector': 'thead th',  
            'props': [('background-color', '#006400'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]
        }, {
            'selector': 'tbody tr:nth-child(even)', 
            'props': [('background-color', '#EFEFEF')]
        }, {
            'selector': 'tbody tr:nth-child(odd)',  
            'props': [('background-color', 'white')]
        }, {
            'selector': 'th.col_heading',  
            'props': [('background-color', '#D3D3D3'), ('color', 'black'), ('font-weight', 'bold'), ('font-style', 'italic')]
        }, {
            'selector': 'tbody td',  
            'props': [('padding', '10px')]  
        }])

    
    st.write(styled_df)




# Visualization 6: 
    st.write("### Correlation between some numerical attributes of the dataset")
    st.write("""In this section, we aim to analyze some numerical variables in the dataset. In the first plot, you can choose between four different variables in the dataset, 
             which will give you some insights into how each one correlates with the others. For example, if we add all of them at the same time, we see that movies with longer 
             durations tend to have lower gross revenue. Conversely, they tend to receive higher (better) IMDb scores. However, in terms of their Metascore, we observe that although 
             many movies with higher IMDb scores also receive higher Metascores, this is not always the case. Furthermore, a similar trend is visible in the second interactive plot, 
             which shows a scatter plot between IMDb scores and Metacritic scores. As a reminder, IMDb scores are derived from ordinary people, who may not necessarily have extensive
              knowledge of movies from a technical standpoint. In contrast, Metacritic scores are typically provided by professional
              film critics, whose opinions tend to be stricter than IMDb ratings. As a result, we observe some movies with good IMDb scores that did not receive high Metascores.""")



    ########################################

   

    # Load the IMDb dataset
    df = pd.read_csv('imdb_1000_final.csv')

    # Sidebar Inputs for attribute selection
    attributes = st.multiselect(
        ' **Select Attributes** to Visualize', 
        ['IMDB_Rating', 'Gross', 'Runtime', 'Meta_score'], 
        default=['IMDB_Rating', 'Gross', 'Runtime']  # Some default attributes selected
    )

    # Drop rows with missing values in selected attributes
    df_selected = df.dropna(subset=attributes)

    if attributes:
        # Parallel Coordinates Plot
        fig = px.parallel_coordinates(
            df_selected,
            color="IMDB_Rating",
            dimensions=attributes,
            labels={col: col.replace('_', ' ') for col in attributes},  # Cleaner axis labels
            color_continuous_scale=px.colors.sequential.Viridis,  # Fancy color scale
            range_color=[df_selected['IMDB_Rating'].min(), df_selected['IMDB_Rating'].max()],
        )
        
        fig.update_layout(
            title={
                'text': '',
                'y': 0.9,  
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=14, family="Verdana", color="black", weight='bold')  
            },
            font=dict(family="Verdana", size=12, color="black"),  
            coloraxis_colorbar=dict(
                title="IMDb Rating",
                titlefont=dict(size=12, weight='bold'),  
                thicknessmode="pixels",
                thickness=20,
                lenmode="fraction",
                len=0.75,
                yanchor="middle",
                y=0.5
            ),
            plot_bgcolor='white',  
            paper_bgcolor='white',  
        )


        fig.update_traces(line=dict(colorbar=dict(thickness=15)))

        st.plotly_chart(fig)

    else:
        st.write("Please select at least one attribute to visualize.")



    ###########################################



    # New Visualization 7: Interactive Scatter Plot (Meta Score vs. IMDB Rating)
    fig = px.scatter(
        movies_df, 
        x='Meta_score', 
        y='IMDB_Rating',
        hover_data=['Series_Title', 'Released_Year'],  
        title=" IMDB Rating vs. Meta Score",  
        labels={'Meta_score': 'Meta Score', 'IMDB_Rating': 'IMDB Rating'},  
        color='IMDB_Rating',  
        color_continuous_scale=px.colors.sequential.Teal  
    )


    fig.update_layout(
        title={
            'text': "🎬 IMDB Rating vs. Meta Score",
            'y': 0.9,  
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, family="Verdana", color="black", weight='bold')  
        },
        font=dict(family="Verdana", size=12, color="black"),  
        xaxis_title=dict(text='Meta Score', font=dict(size=14, weight='bold')),  
        yaxis_title=dict(text='IMDB Rating', font=dict(size=14, weight='bold')),  
        
        
        xaxis=dict(
            tickfont=dict(size=12, weight='bold'),  
        ),
        yaxis=dict(
            tickfont=dict(size=12, weight='bold'),  
        ),
        
        plot_bgcolor='white',  
        paper_bgcolor='white', 
        hoverlabel=dict(font_size=12, font_family="Arial", font_color="black")  
    )

    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.markdown("**💡 Tip:** _Hover over the plane on each point to see its details better!_")
#########################################
    st.write("### Analysis of IMDB Ratings by genre and gross revenue of each movie:")
    st.write("""In the graphs below, you can observe the correlation between IMDb ratings, genre, and gross revenue for each movie. Users can add or remove specific genres to 
    see how these variables change in relation to one another. From the first graph, it is evident that movies in the western genre tend to have higher distributions of good scores
     compared to other genres. Additionally, it can be observed that drama, one of the most popular genres in the film industry, includes several movies with very high ratings from viewers.""")




    # Load the dataset
    data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

    #st.title("IMDb Rating Analysis by Genre")

    # Main filter section
    #st.subheader("Filter Movies by Genre")
    genres = st.multiselect(
        "Select Genres:",
        options=data['Genre_1'].unique(),
        default=data['Genre_1'].unique(),
        key='h1'
    )

    filtered_data = data[data['Genre_1'].isin(genres)]

    # Visualization 1: IMDb Ratings Distribution by Genre
    #st.subheader("IMDb Ratings Distribution by Genre")
    fig = px.box(
        filtered_data,
        x='Genre_1',
        y='IMDB_Rating',
        color='Genre_1',
        title="IMDb Ratings Distribution by Genre",
        labels={'Genre_1': "Genre", 'IMDB_Rating': "IMDb Rating"}
    )
    st.plotly_chart(fig)

    st.write("""In the graph below, we observe the relationship between IMDb ratings and gross revenue for movies with different certificates. An interesting point from this plot is 
    that even movies with very high IMDb ratings did not always achieve the highest gross revenue in theaters. For example, The Shawshank Redemption performed poorly in terms of gross revenue during its initial release, 
    as it did not immediately capture the interest of audiences in cinemas. However, over time, it gained popularity among moviegoers and ultimately achieved high ratings.""")

    # Visualization 2: Gross Revenue vs IMDb Rating
    #st.subheader("Gross Revenue vs IMDb Rating")
    fig2 = px.scatter(
        filtered_data,
        x='IMDB_Rating',
        y='Gross',
        size='No_of_Votes',
        color='Certificate',
        hover_name='Series_Title',
        title="Gross Revenue vs IMDb Rating",
        labels={'IMDB_Rating': 'IMDb Rating', 'Gross': 'Gross Revenue'}
    )
    st.plotly_chart(fig2)

###########################################







# Visualization 8

    st.write("### Analysis based on the movie summary:")
    st.write("""One of the greatest and most important elements of movies is the story behind them. Most people, whether watching movies professionally or simply for leisure, want to see 
    something that is worth their time and money. Similarly, many authors strive to write stories that can attract as wide an audience as possible. For these reasons, it is crucial to analyze 
    the main topics of movies that are considered top-ranked. One popular visualization often used by data scientists for such analyses is the word cloud. Word clouds help reveal the main topics and keywords associated with a collection of movies. 
    By examining a word cloud, we can quickly grasp the general themes of these movies. In the visualization below, larger words indicate those that appear more frequently in the overviews of 
    top-ranked movies. This allows us to identify the recurring themes and topics of these highly rated films at a glance.""")


    # Load dataset
    df = pd.read_csv('imdb_1000_final.csv')

    # Generate Word Cloud
    text = " ".join(df['Overview'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())

###################
    


    # Load the dataset
    @st.cache_data
    def load_data():
        data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')
        data['Overview'] = data['Overview'].fillna('')  # Ensure no missing values in the Overview column
        return data

    data = load_data()

    # Step 2: Sentiment Analysis Function
    @st.cache_data
    def analyze_sentiment(data):
        def sentiment_analysis(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity

        data['Sentiment_Polarity'], data['Sentiment_Subjectivity'] = zip(*data['Overview'].apply(sentiment_analysis))
        return data

    data = analyze_sentiment(data)


    st.write("""In the below graphs we see the distributauion between the Sentiment Polarity with gross revenue and IMDB ratings of the movies. Similar to the previous graphs, we see the same normal distribution
    for both IMDB rating and gross revenue. More specifically, the plot indicates that most movies have a sentiment polarity near 0 (neutral), while IMDB ratings are clustered between 7.5 and 8.5. A few outliers 
    show higher ratings (close to 9), particularly for movies with slightly positive sentiment polarity, suggesting
     a potential weak positive correlation between sentiment polarity and IMDB Rating. However, the overall dispersion implies a diverse distribution with no strong linear relationship. As we mentioned above,
      we see the similar trend in terms of the gross revenue as well.""")

    # Streamlit App Layout
    #st.title("Movie Sentiment Analysis")
    #st.write("Analyze the relationships between sentiment polarity, IMDb ratings, and gross revenue.")

    # Dropdown for selecting the y-axis
    #st.subheader("Select the Metric to Plot Against Sentiment Polarity")
    y_axis = st.selectbox(
        "Choose a metric for the y-axis:",
        options=['IMDB_Rating', 'Gross'],
        index=0
    )

    # Generate scatter plot dynamically based on y-axis selection
    fig = px.scatter(
        data,
        x='Sentiment_Polarity',
        y=y_axis,
        color=y_axis,
        color_continuous_scale='viridis',  # Fixed colorscale
        size_max=15,
        title=f'Sentiment Polarity vs {y_axis}',
        labels={y_axis: y_axis, 'Sentiment_Polarity': 'Sentiment Polarity'}
    )

    # Display the plot
    st.plotly_chart(fig)






#####################################




    



elif menu == "Machine Learning For Prediction":


    st.title("Machine Learning For Prediction")
    st.write("### Overview of this section:")
    st.write("""In this section, we will present some machine learning analysis for predicting the movie success. We devided this section for two categories of people, which first group
                of them are professional movie industry companies that want to invest on new movies and looking for a metric that can help them to get some advice whether it is good to invest
                on the movie with its charactristics such as genre, cetificate, and story. Moreover, the second group of people are those ordinary people that want to use our app to 
                get some knwoledge and advice for choosing a good movie to watch on their free time and also find the good movies based on their mood and their taste of genre and story.""")





    

    @st.cache_data
    def load_data():
        data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')
        data = data.dropna(subset=['Gross', 'IMDB_Rating', 'Genre_1', 'Certificate'])
        return data

    data = load_data()

    # Preprocessing
    scaler_gross = MinMaxScaler(feature_range=(0, 1))
    scaler_IMDB = MinMaxScaler(feature_range=(0, 1))

    data['Gross'] = scaler_gross.fit_transform(data[['Gross']])
    data['IMDB_Rating'] = scaler_IMDB.fit_transform(data[['IMDB_Rating']])

    # Encode categorical features
    Genre_encoder = LabelEncoder()
    data['Genre_1_encoded'] = Genre_encoder.fit_transform(data['Genre_1'])
    genre_to_number = dict(zip(Genre_encoder.classes_, range(len(Genre_encoder.classes_))))

    Certificate_encoder = LabelEncoder()
    data['Certificate_encoded'] = Certificate_encoder.fit_transform(data['Certificate'])
    certificate_to_number = dict(zip(Certificate_encoder.classes_, range(len(Certificate_encoder.classes_))))

    # Feature Selection
    X = data[['Genre_1_encoded', 'Certificate_encoded']].to_numpy()
    y_gross = data['Gross'].to_numpy()
    y_imdb = data['IMDB_Rating'].to_numpy()

    # Train-test split
    X_train_gross, X_test_gross, y_train_gross, y_test_gross = train_test_split(X, y_gross, test_size=0.2, random_state=42)
    X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X, y_imdb, test_size=0.2, random_state=42)

    # Train models
    model_gross = RandomForestRegressor(random_state=42, n_estimators=40, max_depth=3)
    model_gross.fit(X_train_gross, y_train_gross)

    model_imdb = RandomForestRegressor(random_state=42, n_estimators=40, max_depth=3)
    model_imdb.fit(X_train_imdb, y_train_imdb)

    # Predict on test set
    y_pred_gross = model_gross.predict(X_test_gross)
    y_pred_imdb = model_imdb.predict(X_test_imdb)

    # Model evaluation
    mse_gross = mean_squared_error(y_test_gross, y_pred_gross)
    r2_gross = r2_score(y_test_gross, y_pred_gross)

    mse_imdb = mean_squared_error(y_test_imdb, y_pred_imdb)
    r2_imdb = r2_score(y_test_imdb, y_pred_imdb)


    st.title("Gross Revenue Prediction for Movie Companies")

    # Streamlit App
    st.write("### Machine Learning-Based Movie Success Prediction System:")
    st.write("""In this section, we introduce our new prediction system designed to determine the potential success of your movie in the future.
        Using this part of the application, you can select your movie's genre and certificate to predict its likelihood of success. Our model works by normalizing the IMDb score and gross revenue of movies. 
        We then implement a Random Forest machine learning algorithm to predict the gross revenue and IMDB rating of the movie in the future. This feature provides movie companies and industries with an initial estimate
        to assess whether a movie is worth investing in. Additionally, you can adjust the IMDB ratings to specific aspects for the people's opinion about the film. Our evaluation system works with a very small error as you see below.""") 
    
    if st.button("Model Explanation"):

        st.write("""For the Random Forest model, we set random_state=42 to ensure that the results are reproducible each time.
                  Using n_estimators=40 limits the number of decision trees, balancing performance and training time.
                  A max_depth=3 restricts the complexity of each tree to prevent overfitting and maintain simpler models.""")


    st.write(f"Model Performance: Gross Revenue - MSE = {mse_gross:.2f}, R² = {r2_gross:.2f}")
    st.write(f"Model Performance: IMDb Rating - MSE = {mse_imdb:.2f}, R² = {r2_imdb:.2f}")

    # Dropdown menus for user input
    main_genres = sorted(data['Genre_1'].unique())
    certificates = sorted(data['Certificate'].unique())

    main_genre_input = st.selectbox("Select Main Genre (Genre_1)", main_genres, key='main_genre')
    certificate_input = st.selectbox("Select Certificate", certificates, key='certificate')

    if st.button("Predict"):
        # Encode user inputs
        genre = genre_to_number[main_genre_input]
        certificate = certificate_to_number[certificate_input]
        
        # Prepare input for prediction
        input_features = np.array([[genre, certificate]])
        
        # Predictions
        gross_pred = model_gross.predict(input_features)[0]
        imdb_pred = model_imdb.predict(input_features)[0]
        
        # Scale back the predictions
        gross_pred_original = scaler_gross.inverse_transform(np.array([[gross_pred]]))[0, 0]
        imdb_pred_original = scaler_IMDB.inverse_transform(np.array([[imdb_pred]]))[0, 0]
        
        # Display predictions
        st.success(f"The predicted Gross Revenue is: ${gross_pred_original:,.2f}")
        st.success(f"The predicted IMDb Rating is: {imdb_pred_original:.2f}")

###################################


    # Load the dataset
    data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

    st.write("### Prediction for the average gross revenue over the years:")
    st.write("""In this section, we performed an autoregressive analysis to predict the average gross revenue for different movie certificates over the next five years. This allows us to forecast trends in the movie industry based on certification categories.""")

    if st.button("Model Explanation", key='kahsgdt'):
        st.write("""The AutoReg model used in this part predicts historical average gross revenue values (up to three previous years or lag of 3) to predict future revenues. By fitting this model to the filtered and grouped time series data, it identifies how each year’s revenue depends on prior years. 
                    This allows the model to generate forecasts for the next five years, providing insight into potential future trends for selected certificates.""")

    
    data['Released_Year'] = pd.to_numeric(data['Released_Year'], errors='coerce')
    data['Gross'] = pd.to_numeric(data['Gross'], errors='coerce')

    
    data_cleaned = data.dropna(subset=['Released_Year', 'Gross', 'Certificate'])

    
    gross_by_certificate = (
        data_cleaned.groupby(['Certificate', 'Released_Year'])['Gross']
        .mean()
        .reset_index()
    )

    # Function to perform autoregression and generate interactive Plotly graph
    def plot_autoregression(data, certificate):
        # Filter data for the selected certificate
        group_data = data[data['Certificate'] == certificate]
        
        # Sort data by year
        group_data = group_data.sort_values(by='Released_Year')
        
        
        years = group_data['Released_Year'].values
        gross_revenue = group_data['Gross'].values
        
        # Fit an autoregressive model
        model = AutoReg(gross_revenue, lags=3)
        model_fit = model.fit()
        
        # Predict the next 5 years
        future_years = np.arange(years[-1] + 1, years[-1] + 6)
        predictions = model_fit.predict(start=len(gross_revenue), end=len(gross_revenue) + 4)
        
        # Combine actual and predicted data
        future_data = pd.DataFrame({'Released_Year': future_years, 'Gross': predictions})
        
        # Create interactive plot with Plotly
        fig = go.Figure()

        
        fig.add_trace(go.Scatter(
            x=group_data['Released_Year'],
            y=group_data['Gross'],
            mode='lines+markers',
            name='Actual Data',
            line=dict(color='blue')
        ))

        
        fig.add_trace(go.Scatter(
            x=future_data['Released_Year'],
            y=future_data['Gross'],
            mode='lines+markers',
            name='Predicted Data',
            line=dict(dash='dash', color='orange')
        ))

        
        fig.update_layout(
            title=f'Average Gross Revenue with Predictions for Certificate: {certificate}',
            xaxis_title='Year',
            yaxis_title='Average Gross Revenue',
            legend=dict(x=0.1, y=1),
            template='plotly_white',
            hovermode='x'
        )

        
        st.plotly_chart(fig)

    
    certificate_options = gross_by_certificate['Certificate'].unique()
    user_certificate = st.selectbox("Select a Certificate:", certificate_options)

    # Plot for Certificate
    if user_certificate in certificate_options:
        plot_autoregression(gross_by_certificate, user_certificate)



###############################################



    # Load the dataset
    @st.cache_data
    def load_data():
        data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')
        data['Overview'] = data['Overview'].fillna('')
        data['Poster_Link'] = data['Poster_Link'].fillna('')
        return data

    data = load_data()

    # Step 1: TF-IDF Vectorization for All Overviews
    @st.cache_resource
    def vectorize_text(data):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['Overview'])
        return tfidf_vectorizer, tfidf_matrix

    tfidf_vectorizer, tfidf_matrix = vectorize_text(data)

    # Step 2: Define Relevance Labeling Function
    def label_relevance(query, data, tfidf_vectorizer, tfidf_matrix):
        # Convert the query to a vector
        query_vector = tfidf_vectorizer.transform([query])
        # Compute cosine similarity with all overviews
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        # Label as relevant if similarity score > threshold (e.g., 0.2)
        data['Relevance'] = (similarity_scores > 0.2).astype(int)
        return data, similarity_scores

    # Step 3: Define Recommendation Function
    def recommend_movies_rf(query, data, tfidf_vectorizer, tfidf_matrix):
        # Label relevance based on query
        data, similarity_scores = label_relevance(query, data, tfidf_vectorizer, tfidf_matrix)

        # Ensure both classes are present
        if data['Relevance'].nunique() == 1:
            st.warning("The query is too specific, and no relevant movies were found.")
            return pd.DataFrame()

        # Extract features and labels
        structured_features = data[['Runtime', 'Meta_score', 'Gross']].fillna(0)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
        X = pd.concat([tfidf_df, structured_features.reset_index(drop=True)], axis=1)
        X.columns = X.columns.astype(str)
        y = data['Relevance']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train a Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)

        # Predict Relevance Scores
        relevance_scores = rf_model.predict_proba(X)[:, 1]
        data['Relevance_Score'] = relevance_scores

        # Recommend Top Movies
        recommendations = data.sort_values(by='Relevance_Score', ascending=False).head(5)
        return recommendations[['Series_Title', 'Overview', 'Genre_1', 'Relevance_Score']]

    # Step 4: Streamlit App
    #st.title("Movie Recommendation System")
    st.title("Movie Recommendation System for Movie Lovers")
    st.write("### Movie Recommendation System based on words come to your mind:")
    st.write("""In this part of the app, we use the overviews of the top 1000 movies from IMDb to recommend movie titles based on your mood and the keywords you input into our
     machine learning model. You can enter words that come to mind, and our model will suggest movie titles along with a relevance score. We believe this feature can be useful for 
     anyone looking to find a movie to watch during their free time based on the words that come to their mind.

We have used the Random Forest technique to train our model, and we hope this feature can meet our users' needs by helping them find the movies that align with the topics they are
 interested in watching.""")
    if st.button("Model Explanation",key='kahsgdt1'):

        st.write("""We set random_state=42 to ensure consistent, reproducible results each time the model is run.
                  The n_estimators=100 option specifies the number of individual decision trees, balancing accuracy and computational cost.
                     By default, the random forest classifier uses features like bootstrap sampling and random feature selection at splits 
                 to maintain model diversity and reduce overfitting.""")


    st.write("Find movies similar to your query based on descriptions and genres!")

    # User Input
    query = st.text_input("Enter a query (e.g., 'hero saves world')", "")

    if st.button("Get Recommendations"):
        if query.strip():
            recommended_movies = recommend_movies_rf(query, data, tfidf_vectorizer, tfidf_matrix)
            if not recommended_movies.empty:
                st.subheader("Top Recommendations")
                # Create a beautiful table
                st.markdown(
                    recommended_movies.to_html(index=False, escape=False), 
                    unsafe_allow_html=True
                )
            else:
                st.warning("No recommendations found.")
        else:
            st.warning("Please enter a query to get recommendations.")


############################


    # Load the dataset
    data = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

    # Preprocessing
    # Combine genres into a single column
    data['All_Genres'] = data[['Genre_1', 'Genre_2', 'Genre_3']].fillna('').agg(' '.join, axis=1)

    # Combine important features into a single string
    data['Combined_Features'] = (
        data['All_Genres'] + ' ' +
        data['Director'] + ' ' +
        data['Star1'] + ' ' +
        data['Star2'] + ' ' +
        data['Star3'] + ' ' +
        data['Star4']
    )

    # Vectorization
    # Convert text data into numerical features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Combined_Features'])

    # Similarity Calculation
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Recommendation Function
    def recommend_movies_with_details(title, cosine_sim=cosine_sim, data=data):
        # Get the index of the movie that matches the title
        indices = pd.Series(data.index, index=data['Series_Title']).drop_duplicates()
        
        if title not in indices:
            st.error(f"Movie '{title}' not found in the dataset.")
            return
        
        idx = indices[title]

        # Get similarity scores for all movies with the selected movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top 10 similar movies (excluding itself)
        sim_scores = sim_scores[1:11]
        
        # Get movie indices and similarity percentages
        movie_indices = [i[0] for i in sim_scores]
        similarity_percentages = [i[1] * 100 for i in sim_scores]
        
        # Fetch titles, posters, and additional details of the top 10 movies
        recommended_movies = data.iloc[movie_indices][['Series_Title', 'Poster_Link', 'Released_Year', 'Overview']]
        
        # Display the recommended movies and their details
        st.subheader(f"Movies similar to '{title}':")
        for index, (row, similarity) in enumerate(zip(recommended_movies.iterrows(), similarity_percentages)):
            movie = row[1]
            st.markdown(f"### {index + 1}. {movie['Series_Title']} ({movie['Released_Year']})")
            st.markdown(f"**Similarity:** {similarity:.2f}%")
            st.markdown(f"**Overview:** {movie['Overview']}")
            # Display the poster image
            try:
                response = requests.get(movie['Poster_Link'])
                img = Image.open(BytesIO(response.content))
                st.image(img, width=200)
            except Exception as e:
                st.warning("Poster image not available.")

    # Step 5: Streamlit App
    
    st.write("### Movie Recommendation System based on titles:")
    st.write("""In this section, we help users and movie lovers discover similar movie titles from the top-ranked movies on IMDb. Users can select a movie title from the top
     1000 list on IMDb and find similar titles based on factors such as genre, director, and the actors who starred in the movies. One interesting feature of this section is that it also displays the covers of similar movies, giving users a better sense of the recommended films. Additionally, we provide users with 
     a similarity percentage based on the selected movie title from our dropdown menu. To ensure accurate recommendations, we have used the Cosine similarity function to find the best matches 
    for your preferences.""")
    if st.button("Model Explanation",key='kahsgdt1212'):

        st.write("""Our model creates a “Combined_Features” string for each movie, capturing genres, director, and star actors.
                  Using TF-IDF vectorization and cosine similarity, it measures how closely each movie’s features match a selected movie.
                Finally, it returns the top similar titles based on these similarity scores.""")
    st.write("Find similar movies to your favorite one!")

    # Dropdown menu for selecting a movie
    movie_name = st.selectbox("Choose a movie:", data['Series_Title'].unique())

    if st.button("Recommend"):
        recommend_movies_with_details(movie_name)    




########################################################################################################
elif menu == "Conclusion":
    st.title("Conclusions")

    st.write("""In this section, we will discuss the conclusions and key findings from this project. So far, we have provided a detailed, 
             step-by-step explanation of the processes we followed, including discussing the movie industry, introducing our dataset, 
             performing data processing to prepare it for analysis, and presenting our results through interactive plots. 
             These plots give us deeper insights into what is happening in our dataset and, to a broader extent, 
             offer some initial ideas about trends in this fascinating industry. Moreover, we have implemented multiple machine learning techniques to provide users with accurate predictions 
             from both professional and general perspectives. From a professional standpoint, we designed a machine learning model that helps movie companies evaluate whether their movie is likely 
             to succeed or fail based on its genre and certificate. For general users, 
             we offer a recommendation system that assists movie enthusiasts in finding the most suitable movie titles based on keywords from overviews or their favorite movie titles.""")
    
    st.write("""Moreover, our initial goal in te first version of the movie app was to build an application for movie production corporations to use these analyses to make better decisions 
             for capturing a larger share of the movie market. In this regard, we presented our results on the correlations between variables, such as how movie genres, 
             certificates, directors, actors, and overviews can play an important role in increasing income and receiving high scores from both general audiences and professional critics.""")
    
    st.write("""As a result, from a data scientist's point of view, we have found that some emerging genres in the movie industry can significantly increase a movie's gross revenue and capture people's 
             attention over time. This is due to the fact that technological developments in the movie industry have allowed corporations to produce movies in new genres, such as 
             Sci-Fi and adventure, which are novel to audiences. Additionally, we observed how different movie certificates have evolved over time. This provides valuable information 
             for investors in the industry, helping them choose which types of plots are trending at the moment. Furthermore, while implementing machine learning algorithms on our dataset for both
              numerical and textual data, we recognized the significant role that genre and certificate play in determining a movie's success. Additionally, when developing a recommendation system for 
              the general audience, we explored
              various aspects of the textual data from movie overviews. This enabled us to better identify favorite movie titles for users searching for films that align with their preferences.""")
    
    st.write("""In conclusion, by conducting Initial Data Analysis (IDA), Exploratory Data Analysis (EDA), data cleaning, and implementing machine learning techniques on this dataset, we gained valuable 
             insights into applying data science principles to a real-world problem. We hope this application proves useful to a wide range of users, from professionals in the movie industry making 
            investment decisions at major corporations to everyday individuals seeking engaging and informativeinsights about movies. 
            Whether for corporate strategy or simply finding an interesting movie to enjoy with loved ones during free time, this application aims to cater to diverse needs.""")

    
    

    st.markdown("""
        <div style="text-align: center; font-size: 40px; font-weight: bold; font-family: 'Comic Sans MS', cursive, sans-serif; color: #ff6347;">
            We Hope That You Enjoyed Our Movie App!!
        </div>
        """, unsafe_allow_html=True)
    








