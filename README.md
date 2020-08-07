
## Final Project Submission

Please fill out:
* Student name: Max Steele
* Student pace: full time
* Scheduled project review date/time: Friday, August 7 at 6:00 PM
* Instructor name: James Irving
* Blog post URL: https://zero731.github.io/using_violin_and_stripplots_in_combination


# Introduction

   In an alternate universe, Microsoft has decided to start a new movie studio and begin making original movies. However, they are completely new to the movie industry as a company, and have hired me to analyze current industry trends and provide actionable insights into what types of movies they should make. Through my analysis, I will seek to answer the following questions:
1. Which genres are top grossing worldwide and have the highest return on investment?
* Of the top genres, how does movie budget affect metrics of financial success?
* Of the top genres, how does genre influence movie rating?

Because the industry creates such a wide diversity of movies, it seems pertinent to help the company focus on a smaller scope of movie types that can be expected to perform well based on current industry trends. Question 1 aims to accomplish this by finding out which genres have been the highest grossing and produced the best return on investment in the last decade. 

The second question will then help the company determine how much money it should invest in making each movie by examining how movie budget influences worldwide gross earnings and return on investment within each genre. Presumably, being such a large company, Microsoft will be able to budget at least the industry-standard towards each movie. Each movie's budget should be enough to make a quality movie, and it's possible this varies by genre. We also want to avoid overspending on movies that can be made well on a lower budget. 

Question 3 considers a different aspect of what makes a good movie. While the overarching goal is to make money by making movies, Microsoft should also care about movie ratings. We should not assume that movies with higher ratings make more money; however, ratings are arguably still important. A nascent movie studio needs to quickly build a good reputation for itself. Even if the first few movies perform well at the box office and produce sizeable returns on investment, the movie studio could still go under in the longterm if it becomes known for producing mediocre movies. The studio will need to attract some of the best talent in the industry (writers, producers, and actors) to remain viable, especially since so many movie studios are already well established. It is not unreasonable to assume that many big names would not want to become artistically associated with a studio that produces poorly rated movies.

# Data Preparation

The data used to answer the above questions, explore recent trends in the movie industry, and provide recommendations for Microsoft's new movie studio were obtained from IMDb. Two files were downloaded directly from files provided by IMDb (source: https://datasets.imdbws.com/; files: title.basics.tsv.gz and title.ratings.tsv.gz; documentation: https://www.imdb.com/interfaces/). The remainder of the data not directly provided in the files on the website was scraped as part of a collaborative effort. The code used for scraping was written by Sam Stoltenberg - github.com/skelouse, and can be found in the secondary imdb_webscrape notebook.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
## find the files in directory
import os
os.listdir('data_files/')
```




    ['budget_ratings.csv',
     'title.basics.tsv.gz',
     'title.ratings.tsv.gz',
     '.ipynb_checkpoints',
     'budget_ratings_adj_budget.csv']




```python
## import files as a list
import glob
file_list = glob.glob('data_files/*sv*')
file_list
```




    ['data_files/budget_ratings.csv',
     'data_files/title.basics.tsv.gz',
     'data_files/title.ratings.tsv.gz',
     'data_files/budget_ratings_adj_budget.csv']




```python
## test how to adjust file name to serve as an informative key name
file_list[0].split('/')[-1].replace('.csv', '')
```




    'budget_ratings'




```python
## create an empty dictionary data tables from files
TABLES = {}

## loop through my list of files
for file in file_list:
    
    ## if file name ends with .tsv.gz, read and separate by tabs
    try:
        if file.endswith('tsv.gz'):
            df = pd.read_csv(file, sep='\t', encoding = "latin-1", low_memory=False)
            key = file.split('/')[-1].replace('.tsv.gz', '').replace('.',"_")
     
    ## otherwise read the file as comma separated with defaults   
        else:
            df = pd.read_csv(file, low_memory=False, index_col=0)
            key = file.split('/')[-1].replace('.csv', '')
        
    ## if the above raises an error (due to issue with UTF-8 encoding), change from default encoding to latin-1
    ## and read by separating by tabs and set key name based on file name
    except UnicodeDecodeError:
        df = pd.read_csv(file, sep='\t', encoding = "latin-1", low_memory=False)
        key = file.split('/')[-1].replace('.tsv.gz', '').replace('.',"_")
    
    ## add each DataFrame that was read in to the TABLES dict with key based on file name
    TABLES[key] = df
```


```python
TABLES.keys()
```




    dict_keys(['budget_ratings', 'title_basics', 'title_ratings', 'budget_ratings_adj_budget'])




```python
## assign each DataFrame from TABLES dict to its own variable
budget_ratings = TABLES['budget_ratings_adj_budget']
title_basics = TABLES['title_basics']
title_ratings = TABLES['title_ratings']
```

## Query IMDb Data Files


```python
def check_df(df):
    """
    returns a df reporting the dtype, num of null values, 
    % of null values, and num of unique values found in each column
    Source for code: written by James Irving - github.com/jirvingphd
    """
    info = {'dtypes':df.dtypes,
           'null values':df.isna().sum(),
            '% null': np.round((df.isna().sum()/len(df))*100,2),
           'nunique':df.nunique(),}
    return pd.DataFrame(info).head(len(df.columns))


def check_ends(df):
    """ returns dimensions, head, and tail of given df
    """
    return (display(df.shape, df.head(), df.tail()))
```


```python
## check out the shape, head, and tail of the title_basics df
check_ends(title_basics)
## note that '\N' is a placeholder value (as described by the documentation provided by IMDb)
```


    (7023997, 9)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0000001</td>
      <td>short</td>
      <td>Carmencita</td>
      <td>Carmencita</td>
      <td>0</td>
      <td>1894</td>
      <td>\N</td>
      <td>1</td>
      <td>Documentary,Short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0000002</td>
      <td>short</td>
      <td>Le clown et ses chiens</td>
      <td>Le clown et ses chiens</td>
      <td>0</td>
      <td>1892</td>
      <td>\N</td>
      <td>5</td>
      <td>Animation,Short</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0000003</td>
      <td>short</td>
      <td>Pauvre Pierrot</td>
      <td>Pauvre Pierrot</td>
      <td>0</td>
      <td>1892</td>
      <td>\N</td>
      <td>4</td>
      <td>Animation,Comedy,Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0000004</td>
      <td>short</td>
      <td>Un bon bock</td>
      <td>Un bon bock</td>
      <td>0</td>
      <td>1892</td>
      <td>\N</td>
      <td>12</td>
      <td>Animation,Short</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0000005</td>
      <td>short</td>
      <td>Blacksmith Scene</td>
      <td>Blacksmith Scene</td>
      <td>0</td>
      <td>1893</td>
      <td>\N</td>
      <td>1</td>
      <td>Comedy,Short</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7023992</th>
      <td>tt9916848</td>
      <td>tvEpisode</td>
      <td>Episode #3.17</td>
      <td>Episode #3.17</td>
      <td>0</td>
      <td>2010</td>
      <td>\N</td>
      <td>\N</td>
      <td>Action,Drama,Family</td>
    </tr>
    <tr>
      <th>7023993</th>
      <td>tt9916850</td>
      <td>tvEpisode</td>
      <td>Episode #3.19</td>
      <td>Episode #3.19</td>
      <td>0</td>
      <td>2010</td>
      <td>\N</td>
      <td>\N</td>
      <td>Action,Drama,Family</td>
    </tr>
    <tr>
      <th>7023994</th>
      <td>tt9916852</td>
      <td>tvEpisode</td>
      <td>Episode #3.20</td>
      <td>Episode #3.20</td>
      <td>0</td>
      <td>2010</td>
      <td>\N</td>
      <td>\N</td>
      <td>Action,Drama,Family</td>
    </tr>
    <tr>
      <th>7023995</th>
      <td>tt9916856</td>
      <td>short</td>
      <td>The Wind</td>
      <td>The Wind</td>
      <td>0</td>
      <td>2015</td>
      <td>\N</td>
      <td>27</td>
      <td>Short</td>
    </tr>
    <tr>
      <th>7023996</th>
      <td>tt9916880</td>
      <td>tvEpisode</td>
      <td>Horrid Henry Knows It All</td>
      <td>Horrid Henry Knows It All</td>
      <td>0</td>
      <td>2014</td>
      <td>\N</td>
      <td>10</td>
      <td>Animation,Comedy,Family</td>
    </tr>
  </tbody>
</table>
</div>



```python
## only interested in movies, not tv shows, so need to be able to filter out non-movie categories
title_basics['titleType'].value_counts()
```




    tvEpisode       5028746
    short            752362
    movie            556830
    video            273501
    tvSeries         189009
    tvMovie          122764
    tvMiniSeries      32547
    tvSpecial         29144
    videoGame         26038
    tvShort           13056
    Name: titleType, dtype: int64




```python
from pandasql import sqldf

## define function to query DataFrames
pysqldf = lambda q: sqldf(q, globals())
```


```python
## select movie title IDs (tconst) from title_ratings df 
## use the default inner join so that we only get movies that have ratings and don't introduce null values
## join on title_basics df to filter movies by startYear and titleType


## only want movies made 2010 to 2019

q1 = """
SELECT tconst, titleType, primaryTitle, startYear, genres, averageRating 
FROM title_ratings
JOIN title_basics
USING(tconst)
WHERE startYear > 2009 
AND startYear < 2020
AND (titleType = 'movie')
"""
imdb_df = pysqldf(q1)
```


```python
## check out the shape, head, and tail of the new df
check_ends(imdb_df)
```


    (83669, 6)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0016906</td>
      <td>movie</td>
      <td>Frivolinas</td>
      <td>2014</td>
      <td>Comedy,Musical</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0064322</td>
      <td>movie</td>
      <td>The Woman with the Knife</td>
      <td>2010</td>
      <td>Drama,Thriller</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0069049</td>
      <td>movie</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>Drama</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0100275</td>
      <td>movie</td>
      <td>The Wandering Soap Opera</td>
      <td>2017</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0112502</td>
      <td>movie</td>
      <td>Bigfoot</td>
      <td>2017</td>
      <td>Horror,Thriller</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83664</th>
      <td>tt9915790</td>
      <td>movie</td>
      <td>Bobbyr Bondhura</td>
      <td>2019</td>
      <td>Family</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>83665</th>
      <td>tt9916132</td>
      <td>movie</td>
      <td>The Mystery of a Buryat Lama</td>
      <td>2018</td>
      <td>Biography,Documentary,History</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>83666</th>
      <td>tt9916160</td>
      <td>movie</td>
      <td>DrÃ¸mmeland</td>
      <td>2019</td>
      <td>Documentary</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>83667</th>
      <td>tt9916428</td>
      <td>movie</td>
      <td>The Secret of China</td>
      <td>2019</td>
      <td>Adventure,History,War</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>83668</th>
      <td>tt9916538</td>
      <td>movie</td>
      <td>Kuambil Lagi Hatiku</td>
      <td>2019</td>
      <td>Drama</td>
      <td>8.4</td>
    </tr>
  </tbody>
</table>
</div>



```python
## check out the shape, head, and tail of the next DataFrame to be joined with the new IMDb df
check_ends(budget_ratings)
```


    (31696, 5)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>budget</th>
      <th>gross</th>
      <th>ww_gross</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt3564924</td>
      <td>150000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PG-13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt3565112</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TV-14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt3565174</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>191365.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt3565406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt3565434</td>
      <td>40000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NotRated</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>budget</th>
      <th>gross</th>
      <th>ww_gross</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31691</th>
      <td>tt1610492</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NotRated</td>
    </tr>
    <tr>
      <th>31692</th>
      <td>tt1610516</td>
      <td>NaN</td>
      <td>46623.0</td>
      <td>46623.0</td>
      <td>Unrated</td>
    </tr>
    <tr>
      <th>31693</th>
      <td>tt1610525</td>
      <td>NaN</td>
      <td>320725.0</td>
      <td>502518.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>31694</th>
      <td>tt1610996</td>
      <td>70000.0</td>
      <td>NaN</td>
      <td>8555.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>31695</th>
      <td>tt1611038</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>123609.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
## join imdb_df with budget_ratings df on title id (tconst)
## use the default inner join so that we don't introduce null values

q2 = """
SELECT tconst, primaryTitle, startYear, genres, averageRating, budget, ww_gross
FROM imdb_df
JOIN budget_ratings
USING(tconst)
"""
financial_df = pysqldf(q2)
```


```python
## check out null and unique values in the new df
check_df(financial_df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtypes</th>
      <th>null values</th>
      <th>% null</th>
      <th>nunique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tconst</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>30571</td>
    </tr>
    <tr>
      <th>primaryTitle</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>29717</td>
    </tr>
    <tr>
      <th>startYear</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>genres</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>730</td>
    </tr>
    <tr>
      <th>averageRating</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>91</td>
    </tr>
    <tr>
      <th>budget</th>
      <td>float64</td>
      <td>19380</td>
      <td>63.39</td>
      <td>1673</td>
    </tr>
    <tr>
      <th>ww_gross</th>
      <td>float64</td>
      <td>16556</td>
      <td>54.16</td>
      <td>13700</td>
    </tr>
  </tbody>
</table>
</div>




```python
## check out the shape, head, and tail of the new df
check_ends(financial_df)
```


    (30571, 7)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>Drama</td>
      <td>6.8</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0100275</td>
      <td>The Wandering Soap Opera</td>
      <td>2017</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.6</td>
      <td>NaN</td>
      <td>3624.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0112502</td>
      <td>Bigfoot</td>
      <td>2017</td>
      <td>Horror,Thriller</td>
      <td>4.5</td>
      <td>1300000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0116991</td>
      <td>Mariette in Ecstasy</td>
      <td>2019</td>
      <td>Drama</td>
      <td>7.5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0159369</td>
      <td>Cooper and Hemingway: The True Gen</td>
      <td>2013</td>
      <td>Documentary</td>
      <td>7.6</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30566</th>
      <td>tt9914286</td>
      <td>Sokagin Ãocuklari</td>
      <td>2019</td>
      <td>Drama,Family</td>
      <td>6.4</td>
      <td>NaN</td>
      <td>2833.0</td>
    </tr>
    <tr>
      <th>30567</th>
      <td>tt9914942</td>
      <td>La vida sense la Sara Amat</td>
      <td>2019</td>
      <td>Drama</td>
      <td>6.8</td>
      <td>NaN</td>
      <td>59794.0</td>
    </tr>
    <tr>
      <th>30568</th>
      <td>tt9915790</td>
      <td>Bobbyr Bondhura</td>
      <td>2019</td>
      <td>Family</td>
      <td>7.3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30569</th>
      <td>tt9916160</td>
      <td>DrÃ¸mmeland</td>
      <td>2019</td>
      <td>Documentary</td>
      <td>6.6</td>
      <td>230100.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30570</th>
      <td>tt9916428</td>
      <td>The Secret of China</td>
      <td>2019</td>
      <td>Adventure,History,War</td>
      <td>3.5</td>
      <td>10000000.0</td>
      <td>4408165.0</td>
    </tr>
  </tbody>
</table>
</div>


### Missing Values in IMDb DataFrame without Financial Information


```python
## no values are showing up as null values (except for financial info), 
## should check for placeholder values
display(imdb_df.info())
check_df(imdb_df)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 83669 entries, 0 to 83668
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   tconst         83669 non-null  object 
     1   titleType      83669 non-null  object 
     2   primaryTitle   83669 non-null  object 
     3   startYear      83669 non-null  object 
     4   genres         83669 non-null  object 
     5   averageRating  83669 non-null  float64
    dtypes: float64(1), object(5)
    memory usage: 3.8+ MB



    None





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtypes</th>
      <th>null values</th>
      <th>% null</th>
      <th>nunique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tconst</th>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>83669</td>
    </tr>
    <tr>
      <th>titleType</th>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>primaryTitle</th>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>78929</td>
    </tr>
    <tr>
      <th>startYear</th>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>genres</th>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>960</td>
    </tr>
    <tr>
      <th>averageRating</th>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>



The only column with missing values is 'genres'. Since the questions I want to answer focus quite a lot on genre, dropping the 863 records missing genre information out of the total 83,669 is acceptable given that it represents such a small portion of the entire dataset.


```python
## check how many missing values (placeholder '\N's as seen earlier) there are and where they are
for col in imdb_df.columns:
    
    ## show the head of sliced DataFrames where value in each column is '\N'
    display(imdb_df.loc[imdb_df[col]=='\\N'].head())
    
    ## print the name of the column and the length of the dataframe created for each column
    ## (corresponds to total number of missing values for that variable)
    print('{}: {}'.format(col, len(imdb_df.loc[imdb_df[col]=='\\N'])))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    tconst: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    titleType: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    primaryTitle: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    startYear: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>tt0306058</td>
      <td>movie</td>
      <td>Second Coming</td>
      <td>2012</td>
      <td>\N</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>28</th>
      <td>tt0326592</td>
      <td>movie</td>
      <td>The Overnight</td>
      <td>2010</td>
      <td>\N</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>33</th>
      <td>tt0330811</td>
      <td>movie</td>
      <td>Regret Not Speaking</td>
      <td>2011</td>
      <td>\N</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>778</th>
      <td>tt10087946</td>
      <td>movie</td>
      <td>Six Characters in Search of a Play</td>
      <td>2019</td>
      <td>\N</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>786</th>
      <td>tt10091972</td>
      <td>movie</td>
      <td>Demashq Halab</td>
      <td>2019</td>
      <td>\N</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>


    genres: 863


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    averageRating: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



```python
## drop records that contain missing values
imdb_df.drop(imdb_df.loc[imdb_df['genres']=='\\N'].index, inplace=True)

## reset index now that rows have been dropped
imdb_df = imdb_df.reset_index(drop=True)
```


```python
## check to make sure those placeholders are gone
for col in imdb_df.columns:
    
    ## show the head of sliced DataFrames where value in each column is '\\N'
    display(imdb_df.loc[imdb_df[col]=='\\N'].head())
    
    ## print the name of the column and the length of the dataframe created for each column
    ## (corresponds to total number of missing values for that variable)
    print('{}: {}'.format(col, len(imdb_df.loc[imdb_df[col]=='\\N'])))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    tconst: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    titleType: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    primaryTitle: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    startYear: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    genres: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    averageRating: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



```python
## still left with 82806 movies for full imdb_df without financial info
len(imdb_df)
```




    82806




```python
imdb_df.info()
## startYear is a string which may not be conducive to later analyses
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 82806 entries, 0 to 82805
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   tconst         82806 non-null  object 
     1   titleType      82806 non-null  object 
     2   primaryTitle   82806 non-null  object 
     3   startYear      82806 non-null  object 
     4   genres         82806 non-null  object 
     5   averageRating  82806 non-null  float64
    dtypes: float64(1), object(5)
    memory usage: 3.8+ MB



```python
## convert startYear into integers
imdb_df['startYear'] = imdb_df['startYear'].astype(int)
imdb_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 82806 entries, 0 to 82805
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   tconst         82806 non-null  object 
     1   titleType      82806 non-null  object 
     2   primaryTitle   82806 non-null  object 
     3   startYear      82806 non-null  int64  
     4   genres         82806 non-null  object 
     5   averageRating  82806 non-null  float64
    dtypes: float64(1), int64(1), object(4)
    memory usage: 3.8+ MB



```python
## check for other potential placeholder values that stick out as oddly frequent in ea col
for col in imdb_df.columns:
    
    ## show the head of sliced DataFrames where value in each column is '\N'
    display(imdb_df[col].value_counts())
    
## everything appears relatively normal
## the movie titles that appear frequently are short titles that would be easily 
## reused across all kinds of movies and each title id (tconst) only appears once
```


    tt3921768    1
    tt3975298    1
    tt3430498    1
    tt4076194    1
    tt2057441    1
                ..
    tt4577388    1
    tt4025798    1
    tt6206204    1
    tt3311588    1
    tt5335830    1
    Name: tconst, Length: 82806, dtype: int64



    movie    82806
    Name: titleType, dtype: int64



    The Return                                    11
    Broken                                        10
    The Gift                                      10
    Solo                                           9
    Lucky                                          9
                                                  ..
    Lonesdale                                      1
    Goldberg & Eisenberg: Til Death Do Us Part     1
    Every Blessed Day                              1
    Rock and Roll Fuck'n'Lovely                    1
    Four Friends                                   1
    Name: primaryTitle, Length: 78129, dtype: int64



    2017    9301
    2016    9067
    2018    8884
    2015    8779
    2014    8597
    2013    8137
    2019    7900
    2012    7807
    2011    7438
    2010    6896
    Name: startYear, dtype: int64



    Drama                       13004
    Documentary                 12344
    Comedy                       6076
    Horror                       3006
    Comedy,Drama                 2919
                                ...  
    Animation,Comedy,Mystery        1
    Horror,Musical,Romance          1
    Action,Romance,Western          1
    Comedy,Thriller,Western         1
    Action,Crime,Musical            1
    Name: genres, Length: 959, dtype: int64



    6.4     2634
    6.8     2612
    6.2     2516
    6.6     2484
    7.2     2418
            ... 
    1.3       33
    9.8       33
    9.7       30
    10.0      29
    9.9        5
    Name: averageRating, Length: 91, dtype: int64


### Format Genre Information and Columns


```python
## create 'genre_list' col where the genres are contained in a list rather than one long string
imdb_df['genre_list'] = imdb_df['genres'].apply(lambda x: x.split(','))

print(type(imdb_df['genre_list'][0]))
imdb_df['genre_list'][0]
```

    <class 'list'>





    ['Comedy', 'Musical']




```python
## need to create a column for each genre to be filled w/ boolean values based on the genre(s) of ea movie

## join all genres in the 'genres' col into one long string
all_genres_string = ','.join(imdb_df['genres'])

## split that string by commas, use set() to return only unique values, make those values into a list
all_genres_list = list(set(all_genres_string.split(',')))
all_genres_list = all_genres_list
all_genres_list
```




    ['Crime',
     'Talk-Show',
     'Adult',
     'Thriller',
     'Documentary',
     'Fantasy',
     'Horror',
     'Sci-Fi',
     'Sport',
     'History',
     'War',
     'Romance',
     'Game-Show',
     'Mystery',
     'Biography',
     'Drama',
     'Family',
     'Action',
     'News',
     'Adventure',
     'Western',
     'Comedy',
     'Musical',
     'Animation',
     'Music',
     'Reality-TV']




```python
## use ea item in the list to make cols in df and fill that column with boolean if is/is not that genre
for genre in all_genres_list:
    imdb_df[genre] = imdb_df['genres'].str.contains(genre)

imdb_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>genre_list</th>
      <th>Crime</th>
      <th>Talk-Show</th>
      <th>Adult</th>
      <th>...</th>
      <th>Family</th>
      <th>Action</th>
      <th>News</th>
      <th>Adventure</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Musical</th>
      <th>Animation</th>
      <th>Music</th>
      <th>Reality-TV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0016906</td>
      <td>movie</td>
      <td>Frivolinas</td>
      <td>2014</td>
      <td>Comedy,Musical</td>
      <td>5.6</td>
      <td>[Comedy, Musical]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0064322</td>
      <td>movie</td>
      <td>The Woman with the Knife</td>
      <td>2010</td>
      <td>Drama,Thriller</td>
      <td>6.7</td>
      <td>[Drama, Thriller]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0069049</td>
      <td>movie</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>Drama</td>
      <td>6.8</td>
      <td>[Drama]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0100275</td>
      <td>movie</td>
      <td>The Wandering Soap Opera</td>
      <td>2017</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.6</td>
      <td>[Comedy, Drama, Fantasy]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0112502</td>
      <td>movie</td>
      <td>Bigfoot</td>
      <td>2017</td>
      <td>Horror,Thriller</td>
      <td>4.5</td>
      <td>[Horror, Thriller]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
## add a column for the number of genres a movie spans

## new 'genre_count' col is filled with the sum of all T/F values across all the genre cols for that row
## (axis=1 specifies to add across the row rather than down the col)
imdb_df['genre_count'] = imdb_df[all_genres_list].sum(axis=1)

imdb_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>genre_list</th>
      <th>Crime</th>
      <th>Talk-Show</th>
      <th>Adult</th>
      <th>...</th>
      <th>Action</th>
      <th>News</th>
      <th>Adventure</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Musical</th>
      <th>Animation</th>
      <th>Music</th>
      <th>Reality-TV</th>
      <th>genre_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0016906</td>
      <td>movie</td>
      <td>Frivolinas</td>
      <td>2014</td>
      <td>Comedy,Musical</td>
      <td>5.6</td>
      <td>[Comedy, Musical]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0064322</td>
      <td>movie</td>
      <td>The Woman with the Knife</td>
      <td>2010</td>
      <td>Drama,Thriller</td>
      <td>6.7</td>
      <td>[Drama, Thriller]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0069049</td>
      <td>movie</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>Drama</td>
      <td>6.8</td>
      <td>[Drama]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0100275</td>
      <td>movie</td>
      <td>The Wandering Soap Opera</td>
      <td>2017</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.6</td>
      <td>[Comedy, Drama, Fantasy]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0112502</td>
      <td>movie</td>
      <td>Bigfoot</td>
      <td>2017</td>
      <td>Horror,Thriller</td>
      <td>4.5</td>
      <td>[Horror, Thriller]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
## create a new df that explodes each movie record into multiple rows for each genre it falls into
explode_genre = imdb_df.explode('genre_list') 
explode_genre.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>genre_list</th>
      <th>Crime</th>
      <th>Talk-Show</th>
      <th>Adult</th>
      <th>...</th>
      <th>Action</th>
      <th>News</th>
      <th>Adventure</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Musical</th>
      <th>Animation</th>
      <th>Music</th>
      <th>Reality-TV</th>
      <th>genre_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0016906</td>
      <td>movie</td>
      <td>Frivolinas</td>
      <td>2014</td>
      <td>Comedy,Musical</td>
      <td>5.6</td>
      <td>Comedy</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0016906</td>
      <td>movie</td>
      <td>Frivolinas</td>
      <td>2014</td>
      <td>Comedy,Musical</td>
      <td>5.6</td>
      <td>Musical</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0064322</td>
      <td>movie</td>
      <td>The Woman with the Knife</td>
      <td>2010</td>
      <td>Drama,Thriller</td>
      <td>6.7</td>
      <td>Drama</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0064322</td>
      <td>movie</td>
      <td>The Woman with the Knife</td>
      <td>2010</td>
      <td>Drama,Thriller</td>
      <td>6.7</td>
      <td>Thriller</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0069049</td>
      <td>movie</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>Drama</td>
      <td>6.8</td>
      <td>Drama</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



## Financial  DataFrame Missing Values and Data Cleaning

In this section I made several key decisions that will focus the analysis on top earning movies. Movies that did not include genre information were dropped from the dataset. Additionally, after finding that the data are extremely skewed in terms of the distribution of movie budgets and worldwide gross earnings, titles missing values for either of those fields were dropped. Due to the skewedness and distribution of the data, it did not seem appropriate to replace missing values (which represented over 50% of the records for each column) with a measure of central tendency. There was simply too much missing information. A cautious and informative approach would be to focus on titles for which all necessary information is available. Presumably, the more popular and higher grossing movies will tend to have this information available on IMDb. These are the types of movies we want to focus on and learn about.

Following that line of logic, I made the decision to focus on top grossing movies and filtered out movies with worldwide gross earnings below the median of \\$1.65 million. I further filtered the dataset to only include movies with budgets of at least \\$1 million dollars. This was below the median movie budget of \\$5 million dollars, and does not seem like an unreasonable minimum initial investment for a company with as many resources as Microsoft. Additionally focusing on movies with reported budgets of at least $1 million has the added benefit of removing impractical \\$0 budgets which would also impede calculations of ROI ((worldwide gross - budget) / budget).


```python
## inspect summary overview of the other main df to be used for analysis
display(financial_df.info())
financial_df.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30571 entries, 0 to 30570
    Data columns (total 7 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   tconst         30571 non-null  object 
     1   primaryTitle   30571 non-null  object 
     2   startYear      30571 non-null  object 
     3   genres         30571 non-null  object 
     4   averageRating  30571 non-null  float64
     5   budget         11191 non-null  float64
     6   ww_gross       14015 non-null  float64
    dtypes: float64(3), object(4)
    memory usage: 1.6+ MB



    None





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30571.000000</td>
      <td>1.119100e+04</td>
      <td>1.401500e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.136898</td>
      <td>6.332828e+06</td>
      <td>1.585756e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.458011</td>
      <td>2.257107e+07</td>
      <td>8.335009e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.300000</td>
      <td>3.000000e+04</td>
      <td>3.045850e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.300000</td>
      <td>3.000000e+05</td>
      <td>2.828170e+05</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.100000</td>
      <td>2.000000e+06</td>
      <td>2.714502e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>3.560000e+08</td>
      <td>2.797801e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
## add cols to df that represent monetary values in millions of dollars for easier interpretation
financial_df['budget1mil'] = round(financial_df['budget']/1000000, 2)
financial_df['ww_gross1mil'] = round(financial_df['ww_gross']/1000000, 2)
```


```python
# check how many missing values (placeholder '\N's) there are and where they are
for col in financial_df.columns:
    
    ## show the head of sliced DataFrames where value in each column is '\\N'
    display(financial_df.loc[financial_df[col]=='\\N'].head())
    
    ## print the name of the column and the length of the dataframe created for each column
    ## (corresponds to total number of missing values for that variable)
    print('{}: {}'.format(col, len(financial_df.loc[financial_df[col]=='\\N'])))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    tconst: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    primaryTitle: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    startYear: 0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>tt0330811</td>
      <td>Regret Not Speaking</td>
      <td>2011</td>
      <td>\N</td>
      <td>6.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>566</th>
      <td>tt10087946</td>
      <td>Six Characters in Search of a Play</td>
      <td>2019</td>
      <td>\N</td>
      <td>4.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>651</th>
      <td>tt10151496</td>
      <td>The White House: Inside Story</td>
      <td>2016</td>
      <td>\N</td>
      <td>7.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>tt10417498</td>
      <td>Infierno grande</td>
      <td>2019</td>
      <td>\N</td>
      <td>5.9</td>
      <td>NaN</td>
      <td>1690.0</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1121</th>
      <td>tt10443324</td>
      <td>Nuts</td>
      <td>2018</td>
      <td>\N</td>
      <td>6.8</td>
      <td>NaN</td>
      <td>4778869.0</td>
      <td>NaN</td>
      <td>4.78</td>
    </tr>
  </tbody>
</table>
</div>


    genres: 337


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    averageRating: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    budget: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    ww_gross: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    budget1mil: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    ww_gross1mil: 0


    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/pandas/core/ops/array_ops.py:253: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      res_values = method(rvalues)


The only column with missing values is 'genres'. Since the questions I want to answer focus quite a lot on genre, dropping the 337 records missing genre information out of the total 30,571 is acceptable given that it represents such a small portion of the entire dataset.


```python
## drop records that contain those missing values
financial_df.drop(financial_df.loc[financial_df['genres']=='\\N'].index, inplace=True)

## reset index now that rows have been dropped
financial_df = financial_df.reset_index(drop=True)
```


```python
## check out null and unique values in financial_df
check_df(financial_df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtypes</th>
      <th>null values</th>
      <th>% null</th>
      <th>nunique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tconst</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>30234</td>
    </tr>
    <tr>
      <th>primaryTitle</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>29389</td>
    </tr>
    <tr>
      <th>startYear</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>genres</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>729</td>
    </tr>
    <tr>
      <th>averageRating</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>91</td>
    </tr>
    <tr>
      <th>budget</th>
      <td>float64</td>
      <td>19069</td>
      <td>63.07</td>
      <td>1672</td>
    </tr>
    <tr>
      <th>ww_gross</th>
      <td>float64</td>
      <td>16385</td>
      <td>54.19</td>
      <td>13545</td>
    </tr>
    <tr>
      <th>budget1mil</th>
      <td>float64</td>
      <td>19069</td>
      <td>63.07</td>
      <td>655</td>
    </tr>
    <tr>
      <th>ww_gross1mil</th>
      <td>float64</td>
      <td>16385</td>
      <td>54.19</td>
      <td>2554</td>
    </tr>
  </tbody>
</table>
</div>




```python
## set up figure and axes subplots
fig, axes = plt.subplots(1,2, figsize=(15,6))

ax1 = axes[0]
ax2 = axes[1]

## plot histogram showing distribution of movie budgets in millions of dollars
sns.distplot(financial_df['budget1mil'], ax=ax1)
ax1.set(xlim=(-10, 400))
ax1.set_title('Distribution of Movie Budgets', fontsize=16, weight='bold')

## plot lines showing the mean and median movies budgets
ax1.axvline(financial_df['budget1mil'].mean(), c='r', 
           label='mean = {}'.format(round(financial_df['budget1mil'].mean(), 2)))
ax1.axvline(financial_df['budget1mil'].median(), c='g', 
           label='median = {}'.format(round(financial_df['budget1mil'].median(), 2)))

ax1.set_ylabel('Frequency', fontsize=14, weight='bold')
ax1.set_xlabel('Budget ($ millions)', fontsize=12, weight='bold')
ax1.set_xticklabels(ax1.get_xticks(), fontsize=12)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=12)
ax1.legend(fontsize=12)

## annotate subplot with min and max movie budgets
ax1.text(250, 0.04, 'min: ${} mil\nmax: ${} mil'.format(round(financial_df['budget1mil'].min(), 2), 
                                               round(financial_df['budget1mil'].max(), 2)), 
                                               fontsize=12, bbox=dict(boxstyle='round', fc='none'))

## plot histogram showing distribution of worldwide gross earnings in millions of dollars
sns.distplot(financial_df['ww_gross1mil'], ax=ax2)
ax2.set(xlim=(-25, 1000))
ax2.set_title('Distribution of Worldwide Gross Distribution', fontsize=16, weight='bold')

## plot lines showing the mean and median ww gross earnings
ax2.axvline(financial_df['ww_gross1mil'].mean(), c='r', 
           label='mean = {}'.format(round(financial_df['ww_gross1mil'].mean(), 2)))
ax2.axvline(financial_df['ww_gross1mil'].median(), c='g', 
           label='median = {}'.format(round(financial_df['ww_gross1mil'].median(), 2)))

ax2.set_xlabel('Worldwide Gross ($ millions)', fontsize=12, weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), fontsize=12)
ax2.set_yticklabels(ax2.get_yticks(), fontsize=12)
ax2.legend(fontsize=12)

## annotate subplot with min and max ww gross earnings
ax2.text(600, 0.006, 'min: ${} mil\nmax: ${} mil'.format(round(financial_df['ww_gross1mil'].min(), 2), 
                                               round(financial_df['ww_gross1mil'].max(), 2)), 
                                               fontsize=12, bbox=dict(boxstyle='round', fc='none'))
```

    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:20: UserWarning: FixedFormatter should only be used together with FixedLocator
    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:21: UserWarning: FixedFormatter should only be used together with FixedLocator
    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:41: UserWarning: FixedFormatter should only be used together with FixedLocator
    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:42: UserWarning: FixedFormatter should only be used together with FixedLocator





    Text(600, 0.006, 'min: $0.0 mil\nmax: $2797.8 mil')




![png](output_47_2.png)


Based on the histograms above, it is obvious that the data for both movie budget and worldwide gross are extremely skewed. As such, simply replacing all the missing values with a measure of central tendency such as the mean, median, or mode could inadvertently skew the results. The most cautious approach would be to drop rows that are missing information on budget and worldwide gross. Presumably the most popular and profitable movies would tend to have this information reported on IMDb. Those are the types of movies Microsoft would want to make, so it makes sense to focus the scope of our investigation on those types of movies.


```python
## drop all rows with null values for budget or worldwide gross
## these values are necessary to calculate return on investment
financial_dropna = financial_df.dropna()
financial_dropna.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3263 entries, 6 to 30233
    Data columns (total 9 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   tconst         3263 non-null   object 
     1   primaryTitle   3263 non-null   object 
     2   startYear      3263 non-null   object 
     3   genres         3263 non-null   object 
     4   averageRating  3263 non-null   float64
     5   budget         3263 non-null   float64
     6   ww_gross       3263 non-null   float64
     7   budget1mil     3263 non-null   float64
     8   ww_gross1mil   3263 non-null   float64
    dtypes: float64(5), object(4)
    memory usage: 254.9+ KB



```python
financial_dropna.describe()
## if we focus on 3rd and 4th quartile for ww gross, min would become $1.65 mil
## we are most interested in top performing movies
## we should also filter budgets
## being a large company, Microsoft will have resources to invest in making movies
## a budget of $1 mil or greater seems reasonable, and filtering like this would have the added benefit of 
## dropping impractical $0 budgets, allowing calc of ROI for every movie left in the data set
## (can't divide by zero)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3263.000000</td>
      <td>3.263000e+03</td>
      <td>3.263000e+03</td>
      <td>3263.000000</td>
      <td>3263.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.108612</td>
      <td>1.971010e+07</td>
      <td>5.549831e+07</td>
      <td>19.710086</td>
      <td>55.498226</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.134765</td>
      <td>3.812692e+07</td>
      <td>1.630340e+08</td>
      <td>38.126918</td>
      <td>163.033989</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.100000</td>
      <td>0.000000e+00</td>
      <td>1.300000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.400000</td>
      <td>1.200000e+06</td>
      <td>9.162800e+04</td>
      <td>1.200000</td>
      <td>0.090000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.200000</td>
      <td>5.000000e+06</td>
      <td>1.647416e+06</td>
      <td>5.000000</td>
      <td>1.650000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.900000</td>
      <td>2.000000e+07</td>
      <td>3.022341e+07</td>
      <td>20.000000</td>
      <td>30.220000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.500000</td>
      <td>3.560000e+08</td>
      <td>2.797801e+09</td>
      <td>356.000000</td>
      <td>2797.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## select a subset of the financial df without null values
med_financial_df = financial_dropna.loc[
    
    ## subset only contains rows with budgets >= $1 mil
    (financial_dropna['budget1mil']>=1) &
    
    ## and rows with ww_gross >= median
    (financial_dropna['ww_gross1mil']>=financial_dropna['ww_gross1mil'].median())
].reset_index()

## drop 'index' col
med_financial_df.drop(['index'], axis=1, inplace=True)

med_financial_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9.617377e+06</td>
      <td>25.0</td>
      <td>9.62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>1.881333e+08</td>
      <td>90.0</td>
      <td>188.13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0365907</td>
      <td>A Walk Among the Tombstones</td>
      <td>2014</td>
      <td>Action,Crime,Drama</td>
      <td>6.5</td>
      <td>28000000.0</td>
      <td>5.883438e+07</td>
      <td>28.0</td>
      <td>58.83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>7.0</td>
      <td>150000000.0</td>
      <td>1.670401e+09</td>
      <td>150.0</td>
      <td>1670.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0376136</td>
      <td>The Rum Diary</td>
      <td>2011</td>
      <td>Comedy,Drama</td>
      <td>6.2</td>
      <td>45000000.0</td>
      <td>3.013496e+07</td>
      <td>45.0</td>
      <td>30.13</td>
    </tr>
  </tbody>
</table>
</div>




```python
med_financial_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1604 entries, 0 to 1603
    Data columns (total 9 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   tconst         1604 non-null   object 
     1   primaryTitle   1604 non-null   object 
     2   startYear      1604 non-null   object 
     3   genres         1604 non-null   object 
     4   averageRating  1604 non-null   float64
     5   budget         1604 non-null   float64
     6   ww_gross       1604 non-null   float64
     7   budget1mil     1604 non-null   float64
     8   ww_gross1mil   1604 non-null   float64
    dtypes: float64(5), object(4)
    memory usage: 112.9+ KB



```python
med_financial_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1604.000000</td>
      <td>1.604000e+03</td>
      <td>1.604000e+03</td>
      <td>1604.000000</td>
      <td>1604.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.279177</td>
      <td>3.658362e+07</td>
      <td>1.124804e+08</td>
      <td>36.583579</td>
      <td>112.480343</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.009705</td>
      <td>4.849304e+07</td>
      <td>2.183903e+08</td>
      <td>48.493053</td>
      <td>218.390315</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.300000</td>
      <td>1.000000e+06</td>
      <td>1.647416e+06</td>
      <td>1.000000</td>
      <td>1.650000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.700000</td>
      <td>7.000000e+06</td>
      <td>6.790422e+06</td>
      <td>7.000000</td>
      <td>6.787500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.400000</td>
      <td>1.900000e+07</td>
      <td>3.176146e+07</td>
      <td>19.000000</td>
      <td>31.760000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>4.000000e+07</td>
      <td>1.126424e+08</td>
      <td>40.000000</td>
      <td>112.640000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.800000</td>
      <td>3.560000e+08</td>
      <td>2.797801e+09</td>
      <td>356.000000</td>
      <td>2797.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## set up figure and axes subplots
fig, axes = plt.subplots(1,2, figsize=(15,6))

ax1 = axes[0]
ax2 = axes[1]

## plot histogram showing distribution of movie budgets in millions of dollars
sns.distplot(med_financial_df['budget1mil'], ax=ax1)
ax1.set(xlim=(-10, 350))
ax1.set_title('Distribution of Movie Budgets', fontsize=16, weight='bold')

## plot lines showing the mean and median movies budgets
ax1.axvline(med_financial_df['budget1mil'].mean(), c='r', 
           label='mean = {}'.format(round(med_financial_df['budget1mil'].mean(), 2)))
ax1.axvline(med_financial_df['budget1mil'].median(), c='g', 
           label='median = {}'.format(round(med_financial_df['budget1mil'].median(), 2)))

ax1.set_ylabel('Frequency', fontsize=14, weight='bold')
ax1.set_xlabel('Budget ($ millions)', fontsize=12, weight='bold')
ax1.set_xticklabels(ax1.get_xticks(), fontsize=12)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=12)
ax1.legend(fontsize=12)

## annotate subplot with min and max movie budgets
ax1.text(235, 0.0075, 'min: ${} mil\nmax: ${} mil'.format(round(med_financial_df['budget1mil'].min(), 2), 
                                               round(med_financial_df['budget1mil'].max(), 2)), 
                                               fontsize=12, bbox=dict(boxstyle='round', fc='none'))


## plot histogram showing distribution of worldwide gross earnings in millions of dollars
sns.distplot(med_financial_df['ww_gross1mil'], ax=ax2)
ax2.set(xlim=(-50, 1300))
ax2.set_title('Distribution of Worldwide Gross', fontsize=16, weight='bold')

## plot lines showing the mean and median ww gross earnings
ax2.axvline(med_financial_df['ww_gross1mil'].mean(), c='r', 
           label='mean = {}'.format(round(med_financial_df['ww_gross1mil'].mean(), 2)))
ax2.axvline(med_financial_df['ww_gross1mil'].median(), c='g', 
           label='median = {}'.format(round(med_financial_df['ww_gross1mil'].median(), 2)))

ax2.set_xlabel('Worldwide Gross ($ millions)', fontsize=12, weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), fontsize=12)
ax2.set_yticklabels(ax2.get_yticks(), fontsize=12)
ax2.legend(fontsize=12)

## annotate subplot with min and max ww gross earnings
ax2.text(800, 0.002, 'min: ${} mil\nmax: ${} mil'.format(round(med_financial_df['ww_gross1mil'].min(), 2), 
                                               round(med_financial_df['ww_gross1mil'].max(), 2)), 
                                               fontsize=12, bbox=dict(boxstyle='round', fc='none'))
```

    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:20: UserWarning: FixedFormatter should only be used together with FixedLocator
    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:21: UserWarning: FixedFormatter should only be used together with FixedLocator
    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:42: UserWarning: FixedFormatter should only be used together with FixedLocator
    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:43: UserWarning: FixedFormatter should only be used together with FixedLocator





    Text(800, 0.002, 'min: $1.65 mil\nmax: $2797.8 mil')




![png](output_54_2.png)



```python
## create a new column containing the return on investment for each movie
med_financial_df['numeratorROI'] = med_financial_df['ww_gross'] - med_financial_df['budget']
med_financial_df['ROI'] = (med_financial_df['numeratorROI'] / med_financial_df['budget']) * 100
med_financial_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>numeratorROI</th>
      <th>ROI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9.617377e+06</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-1.538262e+07</td>
      <td>-61.530492</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>1.881333e+08</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>9.813332e+07</td>
      <td>109.037024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0365907</td>
      <td>A Walk Among the Tombstones</td>
      <td>2014</td>
      <td>Action,Crime,Drama</td>
      <td>6.5</td>
      <td>28000000.0</td>
      <td>5.883438e+07</td>
      <td>28.0</td>
      <td>58.83</td>
      <td>3.083438e+07</td>
      <td>110.122800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>7.0</td>
      <td>150000000.0</td>
      <td>1.670401e+09</td>
      <td>150.0</td>
      <td>1670.40</td>
      <td>1.520401e+09</td>
      <td>1013.600425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0376136</td>
      <td>The Rum Diary</td>
      <td>2011</td>
      <td>Comedy,Drama</td>
      <td>6.2</td>
      <td>45000000.0</td>
      <td>3.013496e+07</td>
      <td>45.0</td>
      <td>30.13</td>
      <td>-1.486504e+07</td>
      <td>-33.033427</td>
    </tr>
  </tbody>
</table>
</div>




```python
## create 'genre_list' col where the genres are contained in a list rather than one long string
med_financial_df['genre_list'] = med_financial_df['genres'].apply(lambda x: x.split(','))

print(type(med_financial_df['genre_list'][0]))
med_financial_df['genre_list'][0]
```

    <class 'list'>





    ['Adventure', 'Drama', 'Romance']




```python
## need to create a column for each genre to be filled w/ boolean values based on the genre(s) of ea movie

## join all genres in the 'genres' col into one long string
all_genres_string = ','.join(med_financial_df['genres'])

## split that string by commas, use set() to return only unique values, make those values into a list
all_genres_list = list(set(all_genres_string.split(',')))
all_genres_list
```




    ['Crime',
     'Thriller',
     'Documentary',
     'Fantasy',
     'Horror',
     'Sci-Fi',
     'Sport',
     'History',
     'War',
     'Romance',
     'Mystery',
     'Biography',
     'Drama',
     'Family',
     'Action',
     'Adventure',
     'Western',
     'Comedy',
     'Musical',
     'Animation',
     'Music']




```python
## use ea item in the list to make cols in df and fill that column with boolean if is/is not that genre
for genre in all_genres_list:
    med_financial_df[genre] = med_financial_df['genres'].str.contains(genre)
```


```python
## add a column for the number of genres a movie spans

## new 'genre_count' col is filled with the sum of all T/F values across all the genre cols for that row
## (axis=1 specifies to add across the row rather than down the col)
med_financial_df['genre_count'] = med_financial_df[all_genres_list].sum(axis=1)

## convert startYear into integers
med_financial_df['startYear'] = med_financial_df['startYear'].astype(int)

med_financial_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1604 entries, 0 to 1603
    Data columns (total 34 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   tconst         1604 non-null   object 
     1   primaryTitle   1604 non-null   object 
     2   startYear      1604 non-null   int64  
     3   genres         1604 non-null   object 
     4   averageRating  1604 non-null   float64
     5   budget         1604 non-null   float64
     6   ww_gross       1604 non-null   float64
     7   budget1mil     1604 non-null   float64
     8   ww_gross1mil   1604 non-null   float64
     9   numeratorROI   1604 non-null   float64
     10  ROI            1604 non-null   float64
     11  genre_list     1604 non-null   object 
     12  Crime          1604 non-null   bool   
     13  Thriller       1604 non-null   bool   
     14  Documentary    1604 non-null   bool   
     15  Fantasy        1604 non-null   bool   
     16  Horror         1604 non-null   bool   
     17  Sci-Fi         1604 non-null   bool   
     18  Sport          1604 non-null   bool   
     19  History        1604 non-null   bool   
     20  War            1604 non-null   bool   
     21  Romance        1604 non-null   bool   
     22  Mystery        1604 non-null   bool   
     23  Biography      1604 non-null   bool   
     24  Drama          1604 non-null   bool   
     25  Family         1604 non-null   bool   
     26  Action         1604 non-null   bool   
     27  Adventure      1604 non-null   bool   
     28  Western        1604 non-null   bool   
     29  Comedy         1604 non-null   bool   
     30  Musical        1604 non-null   bool   
     31  Animation      1604 non-null   bool   
     32  Music          1604 non-null   bool   
     33  genre_count    1604 non-null   int64  
    dtypes: bool(21), float64(7), int64(2), object(4)
    memory usage: 195.9+ KB



```python
## create a new df that explodes each movie record into multiple rows for each genre it falls into
financial_explode_genre = med_financial_df.explode('genre_list') 
financial_explode_genre.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>numeratorROI</th>
      <th>...</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Musical</th>
      <th>Animation</th>
      <th>Music</th>
      <th>genre_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9617377.0</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-15382623.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9617377.0</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-15382623.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9617377.0</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-15382623.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>188133322.0</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>98133322.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>188133322.0</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>98133322.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



# Q1: Which genres are top grossing worldwide and have the highest return on investment?


```python
## make a dataframe displaying the means for all cols grouped by individual genres
all_genres_means = financial_explode_genre.groupby('genre_list').mean()
all_genres_means
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>startYear</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>numeratorROI</th>
      <th>ROI</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>...</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Musical</th>
      <th>Animation</th>
      <th>Music</th>
      <th>genre_count</th>
    </tr>
    <tr>
      <th>genre_list</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Action</th>
      <td>2014.222958</td>
      <td>6.307506</td>
      <td>6.650728e+07</td>
      <td>2.002771e+08</td>
      <td>66.507241</td>
      <td>200.277196</td>
      <td>1.337699e+08</td>
      <td>165.268791</td>
      <td>0.240618</td>
      <td>0.183223</td>
      <td>...</td>
      <td>0.300221</td>
      <td>0.028698</td>
      <td>1.000000</td>
      <td>0.481236</td>
      <td>0.002208</td>
      <td>0.192053</td>
      <td>0.000000</td>
      <td>0.052980</td>
      <td>0.000000</td>
      <td>2.911700</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>2014.634961</td>
      <td>6.268895</td>
      <td>8.070955e+07</td>
      <td>2.677183e+08</td>
      <td>80.709537</td>
      <td>267.718252</td>
      <td>1.870087e+08</td>
      <td>184.098049</td>
      <td>0.053985</td>
      <td>0.041131</td>
      <td>...</td>
      <td>0.164524</td>
      <td>0.143959</td>
      <td>0.560411</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.349614</td>
      <td>0.000000</td>
      <td>0.277635</td>
      <td>0.002571</td>
      <td>2.958869</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>2014.492063</td>
      <td>6.288889</td>
      <td>6.297082e+07</td>
      <td>2.067320e+08</td>
      <td>62.970873</td>
      <td>206.732143</td>
      <td>1.437612e+08</td>
      <td>179.193330</td>
      <td>0.015873</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.039683</td>
      <td>0.198413</td>
      <td>0.190476</td>
      <td>0.857143</td>
      <td>0.000000</td>
      <td>0.603175</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.928571</td>
    </tr>
    <tr>
      <th>Biography</th>
      <td>2014.711409</td>
      <td>6.878523</td>
      <td>2.075240e+07</td>
      <td>5.217694e+07</td>
      <td>20.752416</td>
      <td>52.176644</td>
      <td>3.142454e+07</td>
      <td>159.208902</td>
      <td>0.127517</td>
      <td>0.040268</td>
      <td>...</td>
      <td>0.939597</td>
      <td>0.046980</td>
      <td>0.053691</td>
      <td>0.040268</td>
      <td>0.000000</td>
      <td>0.107383</td>
      <td>0.006711</td>
      <td>0.000000</td>
      <td>0.060403</td>
      <td>2.778523</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>2014.361386</td>
      <td>6.034323</td>
      <td>3.117767e+07</td>
      <td>9.374557e+07</td>
      <td>31.177624</td>
      <td>93.745462</td>
      <td>6.256790e+07</td>
      <td>186.755062</td>
      <td>0.095710</td>
      <td>0.009901</td>
      <td>...</td>
      <td>0.316832</td>
      <td>0.094059</td>
      <td>0.143564</td>
      <td>0.224422</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.006601</td>
      <td>0.125413</td>
      <td>0.026403</td>
      <td>2.400990</td>
    </tr>
    <tr>
      <th>Crime</th>
      <td>2014.593361</td>
      <td>6.405809</td>
      <td>2.612802e+07</td>
      <td>6.169305e+07</td>
      <td>26.127967</td>
      <td>61.693029</td>
      <td>3.556503e+07</td>
      <td>117.729745</td>
      <td>1.000000</td>
      <td>0.224066</td>
      <td>...</td>
      <td>0.564315</td>
      <td>0.004149</td>
      <td>0.452282</td>
      <td>0.087137</td>
      <td>0.004149</td>
      <td>0.240664</td>
      <td>0.004149</td>
      <td>0.008299</td>
      <td>0.004149</td>
      <td>2.875519</td>
    </tr>
    <tr>
      <th>Documentary</th>
      <td>2013.666667</td>
      <td>7.053333</td>
      <td>8.010759e+06</td>
      <td>2.425956e+07</td>
      <td>8.010667</td>
      <td>24.260000</td>
      <td>1.624880e+07</td>
      <td>228.854907</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>Drama</th>
      <td>2014.377382</td>
      <td>6.608767</td>
      <td>2.388884e+07</td>
      <td>6.941406e+07</td>
      <td>23.888793</td>
      <td>69.413964</td>
      <td>4.552522e+07</td>
      <td>192.999400</td>
      <td>0.172808</td>
      <td>0.115629</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.045743</td>
      <td>0.172808</td>
      <td>0.081321</td>
      <td>0.005083</td>
      <td>0.243964</td>
      <td>0.003812</td>
      <td>0.006353</td>
      <td>0.035578</td>
      <td>2.608640</td>
    </tr>
    <tr>
      <th>Family</th>
      <td>2014.515385</td>
      <td>6.002308</td>
      <td>4.637540e+07</td>
      <td>1.332098e+08</td>
      <td>46.375308</td>
      <td>133.209538</td>
      <td>8.683441e+07</td>
      <td>155.624111</td>
      <td>0.007692</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.276923</td>
      <td>1.000000</td>
      <td>0.100000</td>
      <td>0.430769</td>
      <td>0.000000</td>
      <td>0.438462</td>
      <td>0.007692</td>
      <td>0.192308</td>
      <td>0.030769</td>
      <td>2.753846</td>
    </tr>
    <tr>
      <th>Fantasy</th>
      <td>2014.069930</td>
      <td>6.061538</td>
      <td>6.985330e+07</td>
      <td>2.054231e+08</td>
      <td>69.853287</td>
      <td>205.422937</td>
      <td>1.355698e+08</td>
      <td>176.698210</td>
      <td>0.006993</td>
      <td>0.041958</td>
      <td>...</td>
      <td>0.307692</td>
      <td>0.160839</td>
      <td>0.363636</td>
      <td>0.426573</td>
      <td>0.006993</td>
      <td>0.230769</td>
      <td>0.000000</td>
      <td>0.020979</td>
      <td>0.013986</td>
      <td>2.895105</td>
    </tr>
    <tr>
      <th>History</th>
      <td>2014.306667</td>
      <td>6.689333</td>
      <td>2.128393e+07</td>
      <td>4.162955e+07</td>
      <td>21.283733</td>
      <td>41.629467</td>
      <td>2.034562e+07</td>
      <td>99.343578</td>
      <td>0.066667</td>
      <td>0.053333</td>
      <td>...</td>
      <td>0.866667</td>
      <td>0.000000</td>
      <td>0.160000</td>
      <td>0.053333</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.013333</td>
      <td>2.760000</td>
    </tr>
    <tr>
      <th>Horror</th>
      <td>2014.643836</td>
      <td>5.822603</td>
      <td>1.976034e+07</td>
      <td>8.623638e+07</td>
      <td>19.760411</td>
      <td>86.236438</td>
      <td>6.647605e+07</td>
      <td>698.414031</td>
      <td>0.061644</td>
      <td>0.431507</td>
      <td>...</td>
      <td>0.294521</td>
      <td>0.000000</td>
      <td>0.136986</td>
      <td>0.047945</td>
      <td>0.000000</td>
      <td>0.082192</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.013699</td>
      <td>2.691781</td>
    </tr>
    <tr>
      <th>Music</th>
      <td>2014.243243</td>
      <td>6.475676</td>
      <td>1.778436e+07</td>
      <td>6.001460e+07</td>
      <td>17.784324</td>
      <td>60.014865</td>
      <td>4.223024e+07</td>
      <td>181.735611</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>...</td>
      <td>0.702703</td>
      <td>0.081081</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.324324</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.783784</td>
    </tr>
    <tr>
      <th>Musical</th>
      <td>2014.750000</td>
      <td>6.600000</td>
      <td>3.739428e+07</td>
      <td>1.269011e+08</td>
      <td>37.395000</td>
      <td>126.903750</td>
      <td>8.950686e+07</td>
      <td>122.094166</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.375000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.375000</td>
    </tr>
    <tr>
      <th>Mystery</th>
      <td>2014.620968</td>
      <td>6.176613</td>
      <td>2.029572e+07</td>
      <td>6.991317e+07</td>
      <td>20.295726</td>
      <td>69.913145</td>
      <td>4.961745e+07</td>
      <td>528.302833</td>
      <td>0.201613</td>
      <td>0.435484</td>
      <td>...</td>
      <td>0.475806</td>
      <td>0.008065</td>
      <td>0.088710</td>
      <td>0.040323</td>
      <td>0.000000</td>
      <td>0.072581</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.919355</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>2013.747706</td>
      <td>6.275229</td>
      <td>2.087279e+07</td>
      <td>5.848204e+07</td>
      <td>20.872661</td>
      <td>58.481835</td>
      <td>3.760925e+07</td>
      <td>200.626467</td>
      <td>0.022936</td>
      <td>0.018349</td>
      <td>...</td>
      <td>0.669725</td>
      <td>0.009174</td>
      <td>0.036697</td>
      <td>0.013761</td>
      <td>0.004587</td>
      <td>0.573394</td>
      <td>0.004587</td>
      <td>0.000000</td>
      <td>0.032110</td>
      <td>2.587156</td>
    </tr>
    <tr>
      <th>Sci-Fi</th>
      <td>2014.324074</td>
      <td>6.485185</td>
      <td>9.430721e+07</td>
      <td>3.266203e+08</td>
      <td>94.307222</td>
      <td>326.620463</td>
      <td>2.323131e+08</td>
      <td>237.602878</td>
      <td>0.018519</td>
      <td>0.129630</td>
      <td>...</td>
      <td>0.222222</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.490741</td>
      <td>0.000000</td>
      <td>0.101852</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009259</td>
      <td>2.898148</td>
    </tr>
    <tr>
      <th>Sport</th>
      <td>2014.896552</td>
      <td>6.534483</td>
      <td>1.932466e+07</td>
      <td>3.880321e+07</td>
      <td>19.324483</td>
      <td>38.803793</td>
      <td>1.947855e+07</td>
      <td>105.582448</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.827586</td>
      <td>0.034483</td>
      <td>0.137931</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.206897</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.758621</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>2014.151515</td>
      <td>6.261472</td>
      <td>2.898838e+07</td>
      <td>1.056229e+08</td>
      <td>28.988398</td>
      <td>105.622900</td>
      <td>7.663448e+07</td>
      <td>410.927497</td>
      <td>0.233766</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.393939</td>
      <td>0.000000</td>
      <td>0.359307</td>
      <td>0.069264</td>
      <td>0.000000</td>
      <td>0.025974</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004329</td>
      <td>2.740260</td>
    </tr>
    <tr>
      <th>War</th>
      <td>2014.103448</td>
      <td>6.720690</td>
      <td>1.761116e+07</td>
      <td>2.959039e+07</td>
      <td>17.611034</td>
      <td>29.590345</td>
      <td>1.197923e+07</td>
      <td>41.196197</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.896552</td>
      <td>0.000000</td>
      <td>0.241379</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.689655</td>
    </tr>
    <tr>
      <th>Western</th>
      <td>2014.400000</td>
      <td>6.920000</td>
      <td>3.164000e+07</td>
      <td>6.325883e+07</td>
      <td>31.640000</td>
      <td>63.258000</td>
      <td>3.161883e+07</td>
      <td>131.635531</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.600000</td>
    </tr>
  </tbody>
</table>
<p>21 rows × 30 columns</p>
</div>




```python
financial_explode_genre.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>numeratorROI</th>
      <th>...</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Musical</th>
      <th>Animation</th>
      <th>Music</th>
      <th>genre_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9617377.0</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-15382623.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9617377.0</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-15382623.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9617377.0</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-15382623.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>188133322.0</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>98133322.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>188133322.0</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>98133322.0</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
## calculate mean and median ww gross in million of dollars for all movies
all_mean_wwg1mil = financial_explode_genre['ww_gross1mil'].mean()
all_median_wwg1mil = financial_explode_genre['ww_gross1mil'].median()

## calculate mean and median ROI for all movies
all_mean_ROI = financial_explode_genre['ROI'].mean()
all_median_ROI = financial_explode_genre['ROI'].median()
```


```python
## make a list of genre names to adjust for tick labels
list(financial_explode_genre['genre_list'].unique())
```




    ['Adventure',
     'Drama',
     'Romance',
     'Comedy',
     'Fantasy',
     'Action',
     'Crime',
     'Sci-Fi',
     'Animation',
     'Family',
     'Mystery',
     'History',
     'War',
     'Thriller',
     'Horror',
     'Biography',
     'Sport',
     'Music',
     'Documentary',
     'Western',
     'Musical']




```python
## shorten longer genre names to fix spacing on axis labels
all_genre_xlabels = ['Adventure',
                     'Drama',
                     'Romance',
                     'Comedy',
                     'Fantasy',
                     'Action',
                     'Crime',
                     'Sci-Fi',
                     'Animation',
                     'Family',
                     'Mystery',
                     'History',
                     'War',
                     'Thriller',
                     'Horror',
                     'Bio',
                     'Sport',
                     'Music',
                     'Doc',
                     'Western',
                     'Musical']
```


```python
## make a violin plot showing distribution of ww gross earnings by movie genre for all genres
ax = sns.violinplot(x="genre_list", y="ww_gross1mil", data=financial_explode_genre,
                    inner=None, color=".8")

## plot a strip plot of the same data on top of the violin plot
ax = sns.stripplot(x="genre_list", y="ww_gross1mil", data=financial_explode_genre)

## plot horizontal lines for the mean and median ww gross earnings for all movies
ax.axhline(y=all_mean_wwg1mil, ls='-', c='k', linewidth=3, label='Mean')
ax.axhline(y=all_median_wwg1mil, ls='--', c='k', linewidth=3, label='Median')

ax.set(ylim=(0, 2500))
ax.set_xticklabels(all_genre_xlabels, rotation=50, fontsize=14)
ax.set_xlabel('Movie Genre', fontsize=14, weight='bold')
ax.set_ylabel('Worldwide Gross ($ millions)', fontsize=14, weight='bold')
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.set_title('Distribution of Worldwide Gross by Genre', fontsize=28, weight='bold')
plt.legend(fontsize=16)
ax.get_figure().set_size_inches((12,10))
```


![png](output_67_0.png)



```python
## make a violin plot showing distribution of ROI by movie genre for all genres
ax = sns.violinplot(x="genre_list", y="ROI", data=financial_explode_genre,
                    inner=None, color=".8")

## plot a strip plot of the same data on top of the violin plot
ax = sns.stripplot(x="genre_list", y="ROI", data=financial_explode_genre)

## plot horizontal lines for the mean and median ROI for all movies
ax.axhline(y=all_mean_ROI, ls='-', c='k', linewidth=3, label='Mean')
ax.axhline(y=all_median_ROI, ls='--', c='k', linewidth=3, label='Median')

ax.set(ylim=(-1000, 12000))
ax.set_xticklabels(all_genre_xlabels, rotation=55, fontsize=14)
ax.set_xlabel('Movie Genre', fontsize=14, weight='bold')
ax.set_ylabel('% Return on Investment', fontsize=14, weight='bold')
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.set_title('Distribution of Return on Investment by Genre', fontsize=28, weight='bold')
plt.legend(fontsize=16)
ax.get_figure().set_size_inches((12,10))
```


![png](output_68_0.png)



```python
## create dataframe that summarizes mean values for ea genre
mean_group = financial_explode_genre.groupby('genre_list').mean()

## create dataframe that summarizes median values for ea genre
median_group = financial_explode_genre.groupby('genre_list').median()
```


```python
## find the genres that have mean ww_gross above the median
mean_wwg_topHalf = mean_group.loc[mean_group['ww_gross1mil']>=mean_group['ww_gross1mil'].quantile(0.50)]
mean_wwg_topHalf.index
```




    Index(['Action', 'Adventure', 'Animation', 'Comedy', 'Family', 'Fantasy',
           'Horror', 'Musical', 'Mystery', 'Sci-Fi', 'Thriller'],
          dtype='object', name='genre_list')




```python
## find the genres that have median ww_gross above the median
median_wwg_topHalf = median_group.loc[median_group['ww_gross1mil']>=median_group['ww_gross1mil'].quantile(0.50)]
median_wwg_topHalf.index
```




    Index(['Action', 'Adventure', 'Animation', 'Comedy', 'Family', 'Fantasy',
           'Horror', 'Musical', 'Mystery', 'Sci-Fi', 'Thriller'],
          dtype='object', name='genre_list')




```python
## find the genres that have mean ROI above the median
mean_ROI_topHalf = mean_group.loc[mean_group['ROI']>=mean_group['ROI'].quantile(0.50)]
mean_ROI_topHalf.index
```




    Index(['Adventure', 'Animation', 'Comedy', 'Documentary', 'Drama', 'Horror',
           'Music', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'],
          dtype='object', name='genre_list')




```python
## find the genres that have median ROI above the median
median_ROI_topHalf = median_group.loc[median_group['ROI']>=median_group['ROI'].quantile(0.50)]
median_ROI_topHalf.index
```




    Index(['Action', 'Adventure', 'Animation', 'Comedy', 'Documentary', 'Fantasy',
           'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'],
          dtype='object', name='genre_list')




```python
## make a list of top genres by mean ww_gross
mean_wwg_topHalf_list = list(mean_wwg_topHalf.index)

## make a list of top genres by mean ROI
mean_ROI_topHalf_list = list(mean_ROI_topHalf.index)

## make a list of top genres by ww_gross
median_wwg_topHalf_list = list(median_wwg_topHalf.index)

## make a list of top genres by ROI
median_ROI_topHalf_list = list(median_ROI_topHalf.index)


## add the lists together and pick out the genres that fall in all categories
topH_genres = (mean_wwg_topHalf_list + mean_ROI_topHalf_list +
               median_wwg_topHalf_list + median_ROI_topHalf_list)
topH_counts = pd.Series(topH_genres).value_counts()
topH_counts
```




    Thriller       4
    Sci-Fi         4
    Comedy         4
    Horror         4
    Animation      4
    Mystery        4
    Adventure      4
    Action         3
    Fantasy        3
    Romance        2
    Musical        2
    Family         2
    Documentary    2
    Music          1
    Drama          1
    dtype: int64




```python
## make a list of genres that fall in both categories (top ww gross and top ROI)
top_genre_list = list(topH_counts[:7].index)
print(top_genre_list)
len(top_genre_list)
```

    ['Thriller', 'Sci-Fi', 'Comedy', 'Horror', 'Animation', 'Mystery', 'Adventure']





    7




```python
## give a set order to this list so it doesn't take a random order every time I re-run this notebook
top_genre_list = ['Thriller', 'Comedy', 'Adventure', 'Mystery', 'Horror', 'Animation', 'Sci-Fi']
```


```python
### only add movies to new df that fall into at least one of the top 8 genres
top_genres_df = med_financial_df.loc[(med_financial_df['Adventure'] == True) | 
                                 (med_financial_df['Mystery'] == True) |
                                 (med_financial_df['Animation'] == True) |
                                 (med_financial_df['Sci-Fi'] == True) |
                                 (med_financial_df['Thriller'] == True) |
                                 (med_financial_df['Comedy'] == True) |
                                 (med_financial_df['Horror'] == True) 
                                ]

## check out new top genres df                            
display(top_genres_df.shape, top_genres_df.head())
```


    (1195, 34)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>numeratorROI</th>
      <th>...</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Musical</th>
      <th>Animation</th>
      <th>Music</th>
      <th>genre_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9.617377e+06</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-1.538262e+07</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>1.881333e+08</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>9.813332e+07</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>7.0</td>
      <td>150000000.0</td>
      <td>1.670401e+09</td>
      <td>150.0</td>
      <td>1670.40</td>
      <td>1.520401e+09</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0376136</td>
      <td>The Rum Diary</td>
      <td>2011</td>
      <td>Comedy,Drama</td>
      <td>6.2</td>
      <td>45000000.0</td>
      <td>3.013496e+07</td>
      <td>45.0</td>
      <td>30.13</td>
      <td>-1.486504e+07</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tt0377981</td>
      <td>Gnomeo &amp; Juliet</td>
      <td>2011</td>
      <td>Adventure,Animation,Comedy</td>
      <td>5.9</td>
      <td>36000000.0</td>
      <td>1.939677e+08</td>
      <td>36.0</td>
      <td>193.97</td>
      <td>1.579677e+08</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



```python
## drop genre columns from df that don't correspond to one of the top 8 genres
top_genres_df = top_genres_df.drop(['numeratorROI', 'Music', 'Musical', 'Crime', 'Fantasy', 'Western', 'Family', 
               'Biography', 'History', 'War', 'Documentary', 'Sport', 
               'Romance', 'Drama', 'Horror', 'Action'], axis=1)
top_genres_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>ROI</th>
      <th>genre_list</th>
      <th>Thriller</th>
      <th>Sci-Fi</th>
      <th>Mystery</th>
      <th>Adventure</th>
      <th>Comedy</th>
      <th>Animation</th>
      <th>genre_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9.617377e+06</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-61.530492</td>
      <td>[Adventure, Drama, Romance]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>1.881333e+08</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>109.037024</td>
      <td>[Comedy, Drama, Fantasy]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>7.0</td>
      <td>150000000.0</td>
      <td>1.670401e+09</td>
      <td>150.0</td>
      <td>1670.40</td>
      <td>1013.600425</td>
      <td>[Action, Adventure, Sci-Fi]</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0376136</td>
      <td>The Rum Diary</td>
      <td>2011</td>
      <td>Comedy,Drama</td>
      <td>6.2</td>
      <td>45000000.0</td>
      <td>3.013496e+07</td>
      <td>45.0</td>
      <td>30.13</td>
      <td>-33.033427</td>
      <td>[Comedy, Drama]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tt0377981</td>
      <td>Gnomeo &amp; Juliet</td>
      <td>2011</td>
      <td>Adventure,Animation,Comedy</td>
      <td>5.9</td>
      <td>36000000.0</td>
      <td>1.939677e+08</td>
      <td>36.0</td>
      <td>193.97</td>
      <td>438.799083</td>
      <td>[Adventure, Animation, Comedy]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
## define a function to filter out non-top genres from the genre list column
def fix_genre(genres):
    """ take in a string of genres, split it by commas into a list, and only append genres from 
    top_genre_list to a new list of genres
    Source for code: written by Sam Stoltenberg - github.com/skelouse
    """
    new_genres = []
    genres = genres.split(',')
    for g in genres:
        if g in top_genre_list:
            new_genres.append(g)
    return new_genres
```


```python
## create a new col in the df that filters out all non-top genres from the genre list
top_genres_df['top_genre_list'] = top_genres_df['genres'].apply(lambda x: fix_genre(x))
display(top_genres_df.shape, top_genres_df.head())
```


    (1195, 19)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>ROI</th>
      <th>genre_list</th>
      <th>Thriller</th>
      <th>Sci-Fi</th>
      <th>Mystery</th>
      <th>Adventure</th>
      <th>Comedy</th>
      <th>Animation</th>
      <th>genre_count</th>
      <th>top_genre_list</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9.617377e+06</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-61.530492</td>
      <td>[Adventure, Drama, Romance]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>[Adventure]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>1.881333e+08</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>109.037024</td>
      <td>[Comedy, Drama, Fantasy]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
      <td>[Comedy]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>7.0</td>
      <td>150000000.0</td>
      <td>1.670401e+09</td>
      <td>150.0</td>
      <td>1670.40</td>
      <td>1013.600425</td>
      <td>[Action, Adventure, Sci-Fi]</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>[Adventure, Sci-Fi]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0376136</td>
      <td>The Rum Diary</td>
      <td>2011</td>
      <td>Comedy,Drama</td>
      <td>6.2</td>
      <td>45000000.0</td>
      <td>3.013496e+07</td>
      <td>45.0</td>
      <td>30.13</td>
      <td>-33.033427</td>
      <td>[Comedy, Drama]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>[Comedy]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tt0377981</td>
      <td>Gnomeo &amp; Juliet</td>
      <td>2011</td>
      <td>Adventure,Animation,Comedy</td>
      <td>5.9</td>
      <td>36000000.0</td>
      <td>1.939677e+08</td>
      <td>36.0</td>
      <td>193.97</td>
      <td>438.799083</td>
      <td>[Adventure, Animation, Comedy]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>3</td>
      <td>[Adventure, Animation, Comedy]</td>
    </tr>
  </tbody>
</table>
</div>



```python
## explode the dataframe by this new top_genre_list col to be able to group by only top genres,
 ## not by all genres
explode_top_genres = top_genres_df.explode('top_genre_list') 
explode_top_genres.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>startYear</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>budget1mil</th>
      <th>ww_gross1mil</th>
      <th>ROI</th>
      <th>genre_list</th>
      <th>Thriller</th>
      <th>Sci-Fi</th>
      <th>Mystery</th>
      <th>Adventure</th>
      <th>Comedy</th>
      <th>Animation</th>
      <th>genre_count</th>
      <th>top_genre_list</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>2012</td>
      <td>Adventure,Drama,Romance</td>
      <td>6.1</td>
      <td>25000000.0</td>
      <td>9.617377e+06</td>
      <td>25.0</td>
      <td>9.62</td>
      <td>-61.530492</td>
      <td>[Adventure, Drama, Romance]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>7.3</td>
      <td>90000000.0</td>
      <td>1.881333e+08</td>
      <td>90.0</td>
      <td>188.13</td>
      <td>109.037024</td>
      <td>[Comedy, Drama, Fantasy]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>7.0</td>
      <td>150000000.0</td>
      <td>1.670401e+09</td>
      <td>150.0</td>
      <td>1670.40</td>
      <td>1013.600425</td>
      <td>[Action, Adventure, Sci-Fi]</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>7.0</td>
      <td>150000000.0</td>
      <td>1.670401e+09</td>
      <td>150.0</td>
      <td>1670.40</td>
      <td>1013.600425</td>
      <td>[Action, Adventure, Sci-Fi]</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Sci-Fi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0376136</td>
      <td>The Rum Diary</td>
      <td>2011</td>
      <td>Comedy,Drama</td>
      <td>6.2</td>
      <td>45000000.0</td>
      <td>3.013496e+07</td>
      <td>45.0</td>
      <td>30.13</td>
      <td>-33.033427</td>
      <td>[Comedy, Drama]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
## make a point plot for each top genre showing how ww gross earnings have changed over time
g = sns.catplot(x="startYear", y="ww_gross1mil", col="top_genre_list", col_wrap=4,
                capsize=.2, height=5, aspect=.8, col_order=top_genre_list,
                kind="point", hue="top_genre_list", data=explode_top_genres)

g.set(ylim=(0, 900))

## collapse the array of axes into 1 dimension so it's easy to iterate through
axes = g.axes.flatten()

## iterate through each genre in the list used to order the plots
 ## and assign that genre's name as the title of the subplot with the same index
for i, genre in enumerate(top_genre_list):
    axes[i].set_title(genre, fontsize=16, weight='bold')
    
axes[0].set_ylabel('Worldwide Gross ($ millions)', fontsize='14', weight='bold')
axes[4].set_ylabel('Worldwide Gross ($ millions)', fontsize='14', weight='bold')

for ax in axes[3:]:
    ax.set_xlabel('Year', fontsize='14', weight='bold')

g.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=14)

g.fig.suptitle("Change in Worldwide Gross Over Time By Genre", size=22, weight='bold')
g.fig.subplots_adjust(top=.9)
```


![png](output_82_0.png)



```python
## make a point plot for each top genre showing how ROI has changed over time
g = sns.catplot(x="startYear", y="ROI", col="top_genre_list", col_wrap=4,
                capsize=.2, height=5, aspect=.8, col_order=top_genre_list,
                kind="point", hue="top_genre_list", data=explode_top_genres)

## plot a horizontal line at ROI = 0 to emphasize negative ROIs
g.map(plt.axhline, y=0, ls='--', c='k', linewidth=3).set(ylim=(-100, 3000))

## collapse the array of axes into 1 dimension so it's easy to iterate through
axes = g.axes.flatten()

## iterate through each genre in the list used to order the plots
 ## and assign that genre's name as the title of the subplot with the same index
for i, genre in enumerate(top_genre_list):
    axes[i].set_title(genre, fontsize=16, weight='bold')

axes[0].set_ylabel('% ROI', fontsize='14', weight='bold')
axes[4].set_ylabel('% ROI', fontsize='14', weight='bold')

for ax in axes[3:]:
    ax.set_xlabel('Year', fontsize='14', weight='bold')

g.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=14)

g.fig.suptitle("Change in % ROI Over Time By Genre", size=22, weight='bold')
g.fig.subplots_adjust(top=.9)
```


![png](output_83_0.png)



```python
## calculate mean and median ww gross earnings for only top genres
top_mean_wwg1mil = explode_top_genres['ww_gross1mil'].mean()
top_median_wwg1mil = explode_top_genres['ww_gross1mil'].median()

## calculate mean and median ROI for only top genres
top_mean_ROI = explode_top_genres['ROI'].mean()
top_median_ROI = explode_top_genres['ROI'].median()
```


```python
## make a violin plot showing distribution of ww gross earnings by movie genre for top genres
ax = sns.violinplot(x="top_genre_list", y="ww_gross1mil", order=top_genre_list,
                    data=explode_top_genres,
                    inner=None, color=".8")

## plot a strip plot of the same data on top of the violin plot
ax = sns.stripplot(x='top_genre_list', y="ww_gross1mil", order=top_genre_list,
                data=explode_top_genres, hue="top_genre_list")

## plot horizontal lines for the mean and median ww gross earnings for top genres
ax.axhline(y=top_mean_wwg1mil, ls='-', c='k', linewidth=3, label='Mean')
ax.axhline(y=top_median_wwg1mil, ls='--', c='k', linewidth=3, label='Median')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=14)
ax.set_xlabel('Movie Genre', fontsize=14, weight='bold')
ax.set_ylabel('Worldwide Gross ($ millions)', fontsize=14, weight='bold')
ax.set_yticks(ax.get_yticks())
ax.set(ylim=(0, 3000))
ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.set_title('Distribution of Worldwide Gross for Top Genres', fontsize=28, weight='bold')
plt.legend(fontsize=16, labels=['Mean', 'Median'])
ax.get_figure().set_size_inches((12,10))
```


![png](output_85_0.png)



```python
## make a violin plot showing distribution of ROI by movie genre for top genres
ax = sns.violinplot(x="top_genre_list", y="ROI", order=top_genre_list,
                    data=explode_top_genres,
                    inner=None, color=".8")

## plot a strip plot of the same data on top of the violin plot
ax = sns.stripplot(x='top_genre_list', y="ROI", order=top_genre_list,
                data=explode_top_genres, hue="top_genre_list")

## plot horizontal lines for the mean and median ROI for top genres
ax.axhline(y=top_mean_ROI, ls='-', c='k', linewidth=3, label='Mean')
ax.axhline(y=top_median_ROI, ls='--', c='k', linewidth=3, label='Median')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=14)
ax.set_xlabel('Movie Genre', fontsize=14, weight='bold')
ax.set_ylabel('% ROI', fontsize=14, weight='bold')
ax.set_yticks(ax.get_yticks())
ax.set(ylim=(-1000, 10000))
ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.set_title('Distribution of ROI for Top Genres', fontsize=28, weight='bold')
plt.legend(fontsize=16, labels=['Mean', 'Median'])
ax.get_figure().set_size_inches((12,10))
```


![png](output_86_0.png)


Based on my analysis, I found that movie genres vary widely in terms of both worldwide gross earnings and return on investment. Genres with top worldwide gross earnings that stuck out initially were Adventure, Drama, Action, and Sci-Fi. Return on investment appears to be more consistent within and across genres than worldwide gross earnings. Three genres stuck out clearly as genres where extremely profitable movies can be made on a relatively low budget: Mystery, Thriller, and Horror.

Overall the most consistently high performing genres based on worldwide gross earnings and return on investment are:
- Comedy
* Animation
* Horror
* Adventure
* Thriller
* Mystery
* Sci-Fi

Within this subset of genres, I found that Adventure movies have had the most consistently high worldwide gross earnings since 2010. Animation and Sci-Fi have also had high gross earnings since 2010, but these categories are less consistent than Adventure. In terms of return on investment, Mystery, Thriller, and Horror are all genres in which it is possible to make quite a lot of profit for relatively little investment. However, this trend is not necessarily the norm and appears to vary substantially from year to year. Additionally, while all 7 top genres present relatively low risk of losing money, Animation, Horror, Thriller, and Sci-Fi seem to present the most risk of a negative return on investment. Comedy and Adventure movies have seen steady return on investment since 2010 and seem to present relatively low risk. Even though Animation and Sci-Fi movies appear to be riskier investments, it is worth nothing that both have started to see increasing returns over Adventure and Comedy in the past couple years.

In light of these insights, I would recommend Microsoft's primary focus be on Adventure movies because they have the potential to be extremely high grossing and are most consistent in providing a positive return on investment. It likely would not be wise to produce movies of only a single genre given the diversity of entertainment options available in today's world. As such I would recommend that the studio consider producing some Comedy movies which are also relatively low risk. The studio should also consider producing some Animation and/or Sci-Fi movies as these genres appear to be on the rise. However they are less consistent than the other two genres, so they should not be the main focus of the studio, at least as it starts out.

# Q2: Of the top genres, how does movie budget affect metrics of financial success?


```python
## check out mean budgets of top genres
top_g_means = explode_top_genres.groupby('top_genre_list').mean()
top_g_means['budget1mil']
mean_budget_dict = dict(zip(top_g_means.index, round(top_g_means['budget1mil'],2 )))
mean_budget_dict
```




    {'Adventure': 80.71,
     'Animation': 62.97,
     'Comedy': 31.18,
     'Horror': 19.76,
     'Mystery': 20.3,
     'Sci-Fi': 94.31,
     'Thriller': 28.99}




```python
## check out median budgets of top genres
top_g_medians = explode_top_genres.groupby('top_genre_list').median()
top_g_medians['budget1mil']
median_budget_dict = dict(zip(top_g_medians.index, top_g_medians['budget1mil']))
median_budget_dict
```




    {'Adventure': 60.0,
     'Animation': 39.5,
     'Comedy': 17.0,
     'Horror': 10.0,
     'Mystery': 11.505,
     'Sci-Fi': 74.0,
     'Thriller': 19.0}




```python
## create separate DataFrames for each of the top genres
adventure_df = explode_top_genres.loc[explode_top_genres['top_genre_list']=='Adventure']
mystery_df = explode_top_genres.loc[explode_top_genres['top_genre_list']=='Mystery']
animation_df = explode_top_genres.loc[explode_top_genres['top_genre_list']=='Animation']
scifi_df = explode_top_genres.loc[explode_top_genres['top_genre_list']=='Sci-Fi']
thriller_df = explode_top_genres.loc[explode_top_genres['top_genre_list']=='Thriller']
comedy_df = explode_top_genres.loc[explode_top_genres['top_genre_list']=='Comedy']
horror_df = explode_top_genres.loc[explode_top_genres['top_genre_list']=='Horror']
```


```python
def show_distplot(dataframe, column_name, genre=None, variable=None):
    """ take in a take in a dataframe and the name of a column from that dataframe to plot 
        a seaborn distplot for that variable, with ability to label axes and adjust
        title by providing names for the genre and variable of interest
    """
    ax = sns.distplot(dataframe[column_name])
    ax.set_title('Distribution of {} for {} Movies'.format(variable, genre))
    ax.set_xlabel(variable)
    ax.set_ylabel('Frequency')
    ax.axvline(dataframe[column_name].mean(), c='r', 
               label='Mean {} = {}'.format(variable, round(dataframe[column_name].mean() ,2)))
    ax.axvline(dataframe[column_name].median(), c='g',
               label='Median {} = {}'.format(variable, round(dataframe[column_name].median() ,2)))
    plt.legend()
    plt.show()
```


```python
## plot histograms showing distribution of budgets for ea of the top genres
show_distplot(comedy_df, 'budget1mil', 'Comedy', 'Budget')
show_distplot(horror_df, 'budget1mil', 'Horror', 'Budget')
show_distplot(mystery_df, 'budget1mil', 'Mystery', 'Budget')
show_distplot(animation_df, 'budget1mil', 'Animation', 'Budget')
show_distplot(thriller_df, 'budget1mil', 'Thriller', 'Budget')
show_distplot(adventure_df, 'budget1mil', 'Adventure', 'Budget')
show_distplot(scifi_df, 'budget1mil', 'Sci-Fi', 'Budget')
```


![png](output_93_0.png)



![png](output_93_1.png)



![png](output_93_2.png)



![png](output_93_3.png)



![png](output_93_4.png)



![png](output_93_5.png)



![png](output_93_6.png)



```python
## plot histograms showing distribution of ww gross earnings for ea of the top genres
show_distplot(comedy_df, 'ww_gross1mil', 'Comedy', 'Worldwide Gross')
show_distplot(horror_df, 'ww_gross1mil', 'Horror', 'Worldwide Gross')
show_distplot(mystery_df, 'ww_gross1mil', 'Mystery', 'Worldwide Gross')
show_distplot(animation_df, 'ww_gross1mil', 'Animation', 'Worldwide Gross')
show_distplot(thriller_df, 'ww_gross1mil', 'Thriller', 'Worldwide Gross')
show_distplot(adventure_df, 'ww_gross1mil', 'Adventure', 'Worldwide Gross')
show_distplot(scifi_df, 'ww_gross1mil', 'Sci-Fi', 'Worldwide Gross')
```


![png](output_94_0.png)



![png](output_94_1.png)



![png](output_94_2.png)



![png](output_94_3.png)



![png](output_94_4.png)



![png](output_94_5.png)



![png](output_94_6.png)



```python
## plot a regression line plot for each top genre that shows the relationship
 ## between budget and ROI
g = sns.lmplot(x="budget1mil", y="ROI", col="top_genre_list", hue="top_genre_list", 
               data=explode_top_genres, col_wrap=4, height=4.5, aspect=.7,
               truncate=True, col_order=top_genre_list)
g.set(xlim=(0, 350), ylim=(-2500, 7000))

## plot a horizontal line at ROI = 0 to emphasize negative ROIs
g.map(plt.axhline, y=0, ls='--', c='k', linewidth=3)

## collapse the array of axes into 1 dimension so it's easy to iterate through
axes = g.axes.flatten()

## iterate through each genre in the list used to order the plots
 ## and assign that genre's name as the title of the subplot with the same index
for i, genre in enumerate(top_genre_list):
    axes[i].set_title(genre, fontsize=16, weight='bold')

axes[0].set_ylabel('% ROI', fontsize='14', weight='bold')
axes[4].set_ylabel('% ROI', fontsize='14', weight='bold')

for ax in axes[3:]:
    ax.set_xlabel('Budget ($ millions)', fontsize='14', weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

g.fig.suptitle("Effect of Movie Budget on ROI By Genre", size=22, weight='bold')
g.fig.subplots_adjust(top=.9)
```

    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:24: UserWarning: FixedFormatter should only be used together with FixedLocator



![png](output_95_1.png)



```python
## plot a regression line plot for each top genre that shows the relationship
 ## between budget and ROI
g = sns.lmplot(x="budget1mil", y="ww_gross1mil", col="top_genre_list", hue="top_genre_list", 
               data=explode_top_genres, col_wrap=4, height=4.5, aspect=.7,
               truncate=True, col_order=top_genre_list)
g.set(xlim=(0, 350), ylim=(0, 2000))

## collapse the array of axes into 1 dimension so it's easy to iterate through
axes = g.axes.flatten()

## iterate through each genre in the list used to order the plots
 ## and assign that genre's name as the title of the subplot with the same index
for i, genre in enumerate(top_genre_list):
    axes[i].set_title(genre, fontsize=16, weight='bold')

axes[0].set_ylabel('Worldwide Gross ($ millions)', fontsize='14', weight='bold')
axes[4].set_ylabel('Worldwide Gross ($ millions)', fontsize='14', weight='bold')

for ax in axes[3:]:
    ax.set_xlabel('Budget ($ millions)', fontsize='14', weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

g.fig.suptitle("Effect of Movie Budget on Worldwide Gross By Genre", size=22, weight='bold')
g.fig.subplots_adjust(top=.9)
```

    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:21: UserWarning: FixedFormatter should only be used together with FixedLocator



![png](output_96_1.png)


For most of the top movie genres, a movie's budget does not appear to have a substantial effect on return on investment. Horror, Mystery, and Thriller movies stand out once again in this analysis. These movies have the potential to offer very high returns compared to relatively low investments. However, these genres appear to have a higher risk of negative returns on investment.

All top genres show a similar trend of increasing worldwide gross earnings with increasing movie budgets. The rate of increase (slope of the line) in gross earnings based on increasing movie budgets appears steepest for Adventure, Animation, and Sci-Fi movies. Based on the findings of the previous section, I recommended that the studio focus primarily on Adventure movies. These movies are among some of the most expensive to produce, but they offer consistently high returns and gross earnings compared to the other genres. 

Based on the analysis in this section, I recommend that the budget for Adventure movies be kept between the median for top grossing Adventure movies, \\$60 million, and a maximum of $150 million. Many of the top grossing Adventure movies fall within this range, and ROI does not appear to increase with movie budget beyond \\$150 million. If the studio chooses to focus secondarily on Comedy movies as previously suggested, budgets for those movies can be substantially lower. Budgets between the industry median for top grossing Comedies, \\$17 million and approximately \\$75 million appear to maximize ROI. In the previous section I found that Sci-Fi and Animation movies are generally more risky investments, but they have been performing better over the past few years. If Microsoft chooses to produce Sci-Fi or Animation movies, I recommend the budget for these movies not exceed \\$150 million each as ROI does not appear to increase beyond this level of investment.

# Q3: How does genre influence average movie rating?


```python
## plot a barplot that shows average movie rating by genre
 ## with multiple bars for each genre representing how many total genres the movie spans
ax = sns.barplot(x='top_genre_list', y='averageRating', hue='genre_count', palette='bright',
                 data=explode_top_genres, order=top_genre_list)
ax.set(ylim=(0, 9))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=14)
ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.set_xlabel('Movie Genre', fontsize=14, weight='bold')
ax.set_ylabel('Average Movie Rating', fontsize=14, weight='bold')
ax.set_title('Effect of Number of Genres on Movie Rating', fontsize=20, weight='bold')
ax.legend(title='Number of Genres')
ax.get_figure().set_size_inches((10,7))
```

    /Users/maxsteele/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:7: UserWarning: FixedFormatter should only be used together with FixedLocator
      import sys



![png](output_99_1.png)



```python
## make a point plot for each top genre showing how movie rating has changed over time    
g = sns.catplot(x="startYear", y="averageRating", col="top_genre_list", col_wrap=4,
                capsize=.2, height=4.5, aspect=.7, col_order=top_genre_list, hue="top_genre_list", 
                kind="point", data=explode_top_genres)

## collapse the array of axes into 1 dimension so it's easy to iterate through
axes = g.axes.flatten()

## iterate through each genre in the list used to order the plots
 ## and assign that genre's name as the title of the subplot with the same index
for i, genre in enumerate(top_genre_list):
    axes[i].set_title(genre, fontsize=16, weight='bold')

axes[0].set_ylabel('Average Rating', fontsize='14', weight='bold')
axes[4].set_ylabel('Average Rating', fontsize='14', weight='bold')

for ax in axes[3:]:
    ax.set_xlabel('Year', fontsize='14', weight='bold')

g.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=14)

g.fig.suptitle("Change in Average Rating Over Time By Genre", size=22, weight='bold')
g.fig.subplots_adjust(top=.9)
```


![png](output_100_0.png)


To make the new movie studio sustainable, not only do we need to make a profit off the movies produced, we also need to continue to attract top talent to work within our studio. Writers, producers, and actors may not want to become associated artistically with a studio that regularly produces poorly rated movies, even if they are still able to make a significant amount of money working for the studio. Without enlisting good writers, producers, and actors, the movie studio may struggle to find footing and remain viable over time. 

The results of the analyses in this section lend further support to my recommendation that the studio focus primarily on Adventure and Comedy movies. These movies are less risky financially, and have consistently been rated relatively highly since 2010 (except for Comedy movies in 2014). I have also recommended that the studio consider producing a few Animation and/or Sci-Fi movies to diversify its productions. Both categories are less consistent in terms of ratings when compared to Adventure and Comedy movies. Animation and Sci-Fi movies appear to have greater potential to be rated much higher, but also much lower than Adventure and Comedy movies.

The bulk of my analysis has focused on movie genres individually. This approach has been helpful in narrowing down our focus to a handful of movie types that are currently profitable and popular. However, movies typically fall into more than just one genre. I found that, among movies that fall into at least one top genre category, movies tend to achieve higher ratings when they span 2 to 3 genres, as opposed to only 1 genre. The studio should produce Adventure movies that also span a second genre. Comedy movies should span a total of 2 to 3 genres. Any Animation and Sci-Fi movies produced should span a total of 3 genres. Future work will seek to answer which specific genre combinations are likely to be the most well-received and profitable when focusing primarily on Adventure, Comedy, Animation, and Sci-Fi movies. 

# Conclusion and Recommendations

Because the movie industry is so diverse and many profitable movie studios are already well-established, a new movie studio should not try to compete on all fronts. Rather, Microsoft's new studio should limit its focus to producing quality movies that represent a subset of the total diversity. It also makes sense to focus on types of movies that have been profitable and well received in recent years.

I found that among the top movie genres, Adventure and Comedy movies are the most consistent in achieving high worldwide gross earnings, return on investment, and average movie ratings. I recommend that these types of movies be the studio's primary focus. I also identified Animation and Sci-Fi movies as riskier investments that are still worthy of consideration to diversify the studio's portfolio. The worldwide gross earnings and return on investment of both genres have increased in recent years. Ratings for Animation movies have also been on the rise. My specific recommendations for producing movies within each of these four genres are as follows:

- **Adventure**: This genre should be the primary focus for investment because it most consistently achieves high worldwide gross earnings and average ratings. It also presents lower risk of a negative return on investment, despite typically being made with higher budgets. The budget for each Adventure movie should be kept between the median budget for top grossing Adventure movies of \\$60 million and a maximum of \\$150 million. Adventure movies should also span 2 genres (Adventure and one other) for the best chance of attaining higher ratings.


* **Comedy**: This genre should be the secondary focus for investment because it also consistently achieves relatively high worldwide gross earnings and average ratings. It also presents a low risk of a negative return on investment and can be produced with lower investments than Adventure movies. The budget for each Comedy movie should be kept between the median budget for top grossing Comedy movies of \\$17 million and a maximum of \\$75 million to maximize return on investment. Comedy movies should span 2 to 3 genres (Comedy plus 1 or 2 others) for the best chance of attaining higher ratings.


* **Animation and/or Sci-Fi**: These genres should not be the a main focus for investment because they are associated with more risk. These types of movies can be expensive to make, but the budget for each Animation or Sci-Fi movie should be kept below \\$150 million to minimize risk. These movies should span 3 genres for the best chance of attaining higher ratings.
