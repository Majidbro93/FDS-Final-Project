
# IMDb Top 1000 Movies Data Analysis Project

## Project Overview

 This project is an analysis and visualization of the IMDb Top 1000 movies dataset, which contains various details about some of the most popular movies of all time. The analysis focuses on key factors like gross income, IMDb ratings, Metascores, movie certificates, and genres. By using data science techniques, this project provides insights into trends in the movie industry and aims to help production companies make better decisions based on audience preferences and revenue potential. Moreover, in terms of the technical data process point of view, this project used different imputation techniques for finding the missing values in numerical and categorical variables in the dataset.

### Objectives
- Conduct Initial Data Analysis (IDA) and Exploratory Data Analysis (EDA).
- Perform data cleaning, handling missing values, and standardizing data.
- Develop visualizations to explore relationships between features.
- Use machine learning techniques for prediction and recommendation systems.

## File Descriptions

### Notebooks
1. **Final_project_app.ipynb**: Jupyter Notebook containing the complete workflow for data processing, visualization, and analysis.
2. **Final_project_app_Majid.ipynb**: An alternative notebook with additional modifications and updates.

### Python Script
3. **Movies_project_streamlit_final_version.py**: The main Streamlit app for interactive visualizations and analyses.

### Datasets
4. **imdb_top_1000.csv**: Original dataset downloaded from Kaggle.
5. **imdb_1000_final.csv**: Final dataset after imputations and cleaning.
6. **imdb_1000_final_with_correct_certificate.csv**: Dataset with corrected movie certificate standards.
7. **imdb_enriched_with_sentiments.csv**: Dataset enriched with sentiment analysis on movie overviews.
8. **imdb_mapped_cert_Final.csv**: Intermediate dataset after mapping certificate standards.
9. **imdb_mapped_cert_Final_modified.csv**: Dataset after dropping ambiguous certificates.

### Additional Outputs
10. Various intermediate datasets used for analysis and model training.

## Requirements

- Python 3.8 or above
- Libraries: pandas, numpy, matplotlib, seaborn, plotly, streamlit, sklearn, textblob, wordcloud

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository.
2. Navigate to the project directory:
   ```bash
   cd imdb-analysis-project
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run Movies_project_streamlit_final_version.py
   ```
4. Open the provided URL in your browser to access the app.

## Key Features

### Data Processing
- Detailed data cleaning and handling of missing values using techniques like median imputation and KNN.
- Standardized certificates and genres for better consistency.

### Visualizations
- Analysis of genres over decades.
- Insights into movie certificates and their distribution.
- Correlation between IMDb scores, Metascores, gross revenue, and runtime.
- Sentiment analysis of movie overviews.

### Machine Learning
- Movie recommendation system based on overview keywords.
- Prediction of missing values in categorical columns using Random Forest.

### Interactive Features
- Select genres, directors, and actors to explore trends and distributions.
- Visualizations for sentiment polarity and subjectivity of movie overviews.

## Dataset Details

The primary dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows). It contains the following columns:
- **Poster_Link**: URL to the movie poster.
- **Series_Title**: Movie title.
- **Released_Year**: Year of release.
- **Certificate**: Age-appropriate viewing classification.
- **Runtime**: Movie duration in minutes.
- **Genre**: Movie genres.
- **IMDB_Rating**: IMDb rating score.
- **Overview**: Brief summary of the movie.
- **Meta_score**: Metacritic score.
- **Gross**: Box office gross revenue.

## Acknowledgments

- **Data Source**: Kaggle IMDb Dataset
- **Tools Used**: Python, Jupyter, Streamlit, Plotly, and Seaborn

For any questions or feedback, please contact the project author.
