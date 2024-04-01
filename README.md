Group Project - G4H COMP 79 - Emerging Technologies

Group Members - Arushi (4356299) & Armaan Kaushal (4355849)

Title: Exploring COVID-19 Dataset: Insights and Trends
Description:

The project's goal is to investigate a dataset that includes COVID-19-related data in order to learn more about the pandemic's patterns, effects, and transmission. Data from several locations and eras are included in the collection, including confirmed cases, fatalities, recoveries, testing rates, vaccine progress, and demographic information.

Detailed Project Description:
To derive significant trends and insights, this project will entail a detailed examination and analysis of the COVID-19 dataset. Principal goals consist of:

1.Descriptive Analysis: To comprehend the global distribution of COVID-19 cases, fatalities, and recoveries, do descriptive analysis.

2.Temporal Trends: Examine the temporal trends to find patterns, such as peaks, troughs, and seasonal fluctuations, in the COVID-19 outbreak across time.

3.Utilizing interactive visualizations and maps, do geospatial analysis to see how COVID-19 cases, fatalities, and vaccination rates vary geographically.

4.Demographic Insights: Examine demographic variables like age, gender, and ethnicity to see how COVID-19 affects various population groups.

5.Analyze correlations: Examine relationships between COVID-19 measurements and other elements, including socioeconomic indicators, healthcare capacity, government actions, and environmental variables.

6.Vaccination Progress: Examine vaccination data to monitor immunization programs' advancement, spot variations in vaccination rates, and evaluate their influence on lowering COVID-19 severity and transmission.

Project Outcomes:
Comprehensive insights into the global and regional dynamics of the COVID-19 pandemic. Identification of key trends, patterns, and correlations related to COVID-19 spread, impact, and mitigation efforts. Data-driven recommendations for policymakers, healthcare professionals, and the general public to inform decision-making and response strategies.

Specification of Modifications/New Additions:
Integration of real-time data sources to provide up-to-date information on COVID-19 metrics. Implementation of advanced visualization techniques, such as animated charts, 3D plots, and dashboard interfaces, to enhance data exploration and presentation. Incorporation of machine learning models for predictive analysis and forecasting of COVID-19 trends based on historical data and external factors.

Criteria-Specific Cell:
1.Application and Relevance: Demonstrate the relevance of COVID-19 data exploration in understanding the ongoing pandemic, informing public health strategies, and guiding policy decisions.

2.Data Manipulation and Analysis: Showcase proficiency in data manipulation, analysis, visualization, and interpretation using Python libraries such as pandas, NumPy, Matplotlib, Seaborn, and Plotly.

3.Innovative Visualization Approaches: Utilize innovative visualization approaches to present insights and trends in an engaging and informative manner, catering to diverse stakeholders and audience groups.

[ ]
pip install gdown

Requirement already satisfied: gdown in /usr/local/lib/python3.10/dist-packages (4.7.3)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from gdown) (3.13.3)
Requirement already satisfied: requests[socks] in /usr/local/lib/python3.10/dist-packages (from gdown) (2.31.0)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from gdown) (1.16.0)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from gdown) (4.66.2)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from gdown) (4.12.3)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->gdown) (2.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (2024.2.2)
Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (1.7.1)
[ ]
import pandas as pd

path_to_csv_file = 'https://drive.google.com/file/d/1FznyfErKN49BKQXQKiZDzdypbw9f2RPV/view?usp=drive_link/COVID19_time_series_data.csv'

[ ]
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
[ ]
from google.colab import files
uploaded = files.upload()
# https://drive.google.com/file/d/1FznyfErKN49BKQXQKiZDzdypbw9f2RPV/view?usp=drive_link

[ ]
df = pd.read_csv("time_series_covid19_confirmed_global.csv")
df

Data Loading and Preprocessing
To begin the analysis, we first load the COVID-19 dataset into a pandas DataFrame. The dataset contains information about confirmed COVID-19 cases across different regions and time periods. We use the read_csv() function from the pandas library to read the dataset from the provided link. Let's take a look at the structure of the dataset:

# Import necessary libraries
import pandas as pd

# Load the COVID-19 dataset
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# Display the DataFrame
print(df)
Double-click (or enter) to edit

Importing the library and Uploading the dataset
[ ]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
[ ]
timeseries_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
Data Loading and Preprocessing
To begin the analysis, we first load the COVID-19 dataset into a pandas DataFrame. The dataset contains information about confirmed COVID-19 cases across different regions and time periods. We use the read_csv() function from the pandas library to read the dataset from the provided link. Let's take a look at the structure of the dataset:

[ ]
# Import necessary libraries
import pandas as pd

# Load the COVID-19 dataset
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# Display the first few rows of the dataset
print("First few rows of the COVID-19 dataset:")
print(df.head())

First few rows of the COVID-19 dataset:
  Province/State Country/Region       Lat       Long  1/22/20  1/23/20  \
0            NaN    Afghanistan  33.93911  67.709953        0        0   
1            NaN        Albania  41.15330  20.168300        0        0   
2            NaN        Algeria  28.03390   1.659600        0        0   
3            NaN        Andorra  42.50630   1.521800        0        0   
4            NaN         Angola -11.20270  17.873900        0        0   

   1/24/20  1/25/20  1/26/20  1/27/20  ...  2/28/23  3/1/23  3/2/23  3/3/23  \
0        0        0        0        0  ...   209322  209340  209358  209362   
1        0        0        0        0  ...   334391  334408  334408  334427   
2        0        0        0        0  ...   271441  271448  271463  271469   
3        0        0        0        0  ...    47866   47875   47875   47875   
4        0        0        0        0  ...   105255  105277  105277  105277   

   3/4/23  3/5/23  3/6/23  3/7/23  3/8/23  3/9/23  
0  209369  209390  209406  209436  209451  209451  
1  334427  334427  334427  334427  334443  334457  
2  271469  271477  271477  271490  271494  271496  
3   47875   47875   47875   47875   47890   47890  
4  105277  105277  105277  105277  105288  105288  

[5 rows x 1147 columns]
Summary statistics
It provides a concise overview of the numerical features in a dataset, including measures of central tendency, dispersion, and shape of the distribution. These statistics help in understanding the distribution and characteristics of the data.

[ ]
# Calculate summary statistics for numerical columns
summary_statistics = df.describe()

# Display the summary statistics
print("Summary statistics for numerical columns:")
print(summary_statistics)

Summary statistics for numerical columns:
              Lat        Long     1/22/20     1/23/20     1/24/20     1/25/20  \
count  287.000000  287.000000  289.000000  289.000000  289.000000  289.000000   
mean    19.718719   22.182084    1.927336    2.273356    3.266436    4.972318   
std     25.956609   77.870931   26.173664   26.270191   32.707271   45.523871   
min    -71.949900 -178.116500    0.000000    0.000000    0.000000    0.000000   
25%      4.072192  -32.823050    0.000000    0.000000    0.000000    0.000000   
50%     21.512583   20.939400    0.000000    0.000000    0.000000    0.000000   
75%     40.401784   89.224350    0.000000    0.000000    0.000000    0.000000   
max     71.706900  178.065000  444.000000  444.000000  549.000000  761.000000   

           1/26/20      1/27/20      1/28/20      1/29/20  ...       2/28/23  \
count   289.000000   289.000000   289.000000   289.000000  ...  2.890000e+02   
mean      7.335640    10.134948    19.307958    21.346021  ...  2.336755e+06   
std      63.623197    85.724481   210.329649   211.628535  ...  8.506608e+06   
min       0.000000     0.000000     0.000000     0.000000  ...  0.000000e+00   
25%       0.000000     0.000000     0.000000     0.000000  ...  1.456700e+04   
50%       0.000000     0.000000     0.000000     0.000000  ...  1.032480e+05   
75%       0.000000     0.000000     0.000000     0.000000  ...  1.051998e+06   
max    1058.000000  1423.000000  3554.000000  3554.000000  ...  1.034435e+08   

             3/1/23        3/2/23        3/3/23        3/4/23        3/5/23  \
count  2.890000e+02  2.890000e+02  2.890000e+02  2.890000e+02  2.890000e+02   
mean   2.337519e+06  2.338173e+06  2.338805e+06  2.338992e+06  2.339187e+06   
std    8.511285e+06  8.514488e+06  8.518031e+06  8.518408e+06  8.518645e+06   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    1.456700e+04  1.456700e+04  1.456700e+04  1.456700e+04  1.456700e+04   
50%    1.032480e+05  1.032480e+05  1.032480e+05  1.032480e+05  1.032480e+05   
75%    1.052122e+06  1.052247e+06  1.052382e+06  1.052519e+06  1.052664e+06   
max    1.035339e+08  1.035898e+08  1.036487e+08  1.036508e+08  1.036470e+08   

             3/6/23        3/7/23        3/8/23        3/9/23  
count  2.890000e+02  2.890000e+02  2.890000e+02  2.890000e+02  
mean   2.339387e+06  2.339839e+06  2.340460e+06  2.341073e+06  
std    8.519346e+06  8.521641e+06  8.524968e+06  8.527765e+06  
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  
25%    1.456700e+04  1.456700e+04  1.456700e+04  1.456700e+04  
50%    1.032480e+05  1.032480e+05  1.032480e+05  1.032480e+05  
75%    1.052664e+06  1.052926e+06  1.053068e+06  1.053213e+06  
max    1.036555e+08  1.036909e+08  1.037558e+08  1.038027e+08  

[8 rows x 1145 columns]
[ ]
# Import necessary libraries
import pandas as pd

# Load the COVID-19 dataset
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# Calculate summary statistics for numerical columns
summary_statistics = df.describe()

# Display the summary statistics
print("Summary statistics for numerical columns:")
print(summary_statistics)

Summary statistics for numerical columns:
              Lat        Long     1/22/20     1/23/20     1/24/20     1/25/20  \
count  287.000000  287.000000  289.000000  289.000000  289.000000  289.000000   
mean    19.718719   22.182084    1.927336    2.273356    3.266436    4.972318   
std     25.956609   77.870931   26.173664   26.270191   32.707271   45.523871   
min    -71.949900 -178.116500    0.000000    0.000000    0.000000    0.000000   
25%      4.072192  -32.823050    0.000000    0.000000    0.000000    0.000000   
50%     21.512583   20.939400    0.000000    0.000000    0.000000    0.000000   
75%     40.401784   89.224350    0.000000    0.000000    0.000000    0.000000   
max     71.706900  178.065000  444.000000  444.000000  549.000000  761.000000   

           1/26/20      1/27/20      1/28/20      1/29/20  ...       2/28/23  \
count   289.000000   289.000000   289.000000   289.000000  ...  2.890000e+02   
mean      7.335640    10.134948    19.307958    21.346021  ...  2.336755e+06   
std      63.623197    85.724481   210.329649   211.628535  ...  8.506608e+06   
min       0.000000     0.000000     0.000000     0.000000  ...  0.000000e+00   
25%       0.000000     0.000000     0.000000     0.000000  ...  1.456700e+04   
50%       0.000000     0.000000     0.000000     0.000000  ...  1.032480e+05   
75%       0.000000     0.000000     0.000000     0.000000  ...  1.051998e+06   
max    1058.000000  1423.000000  3554.000000  3554.000000  ...  1.034435e+08   

             3/1/23        3/2/23        3/3/23        3/4/23        3/5/23  \
count  2.890000e+02  2.890000e+02  2.890000e+02  2.890000e+02  2.890000e+02   
mean   2.337519e+06  2.338173e+06  2.338805e+06  2.338992e+06  2.339187e+06   
std    8.511285e+06  8.514488e+06  8.518031e+06  8.518408e+06  8.518645e+06   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    1.456700e+04  1.456700e+04  1.456700e+04  1.456700e+04  1.456700e+04   
50%    1.032480e+05  1.032480e+05  1.032480e+05  1.032480e+05  1.032480e+05   
75%    1.052122e+06  1.052247e+06  1.052382e+06  1.052519e+06  1.052664e+06   
max    1.035339e+08  1.035898e+08  1.036487e+08  1.036508e+08  1.036470e+08   

             3/6/23        3/7/23        3/8/23        3/9/23  
count  2.890000e+02  2.890000e+02  2.890000e+02  2.890000e+02  
mean   2.339387e+06  2.339839e+06  2.340460e+06  2.341073e+06  
std    8.519346e+06  8.521641e+06  8.524968e+06  8.527765e+06  
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  
25%    1.456700e+04  1.456700e+04  1.456700e+04  1.456700e+04  
50%    1.032480e+05  1.032480e+05  1.032480e+05  1.032480e+05  
75%    1.052664e+06  1.052926e+06  1.053068e+06  1.053213e+06  
max    1.036555e+08  1.036909e+08  1.037558e+08  1.038027e+08  

[8 rows x 1145 columns]
Visualizing COVID-19 Cases Over Time
To better understand the spread of COVID-19, we need to analyze the trend of confirmed cases over time. In this section, we'll focus on visualizing the distribution of confirmed COVID-19 cases in specific regions. We'll use line plots to illustrate the progression of cases from the starting date of the outbreak.

Steps: Data Preparation:

We've obtained a dataset containing information about COVID-19 cases in various provinces of China, including the number of confirmed cases for each date starting from 1/22/20. Selecting a Province:

We chose a specific province, in this case, "Hubei," which was the epicenter of the outbreak, to analyze the trend of confirmed cases. Data Extraction:

We extracted the relevant data for the selected province, including dates and the corresponding number of confirmed cases. Plotting the Data:

Using matplotlib, we created a line plot to visualize the trend of confirmed COVID-19 cases over time. The x-axis represents the dates, and the y-axis represents the number of confirmed cases. Each point on the line plot represents the number of confirmed cases on a specific date. Interpretation:

By examining the line plot, we can observe how the number of confirmed cases has evolved over time in the selected province. This visualization helps us understand the trajectory of the outbreak and identify periods of rapid growth or stabilization.

[ ]
import matplotlib.pyplot as plt

# Data for Hubei province
dates = ['1/22/20', '1/23/20', '1/24/20', '1/25/20', '1/26/20', '1/27/20']
cases = [444, 444, 549, 761, 1058, 1423]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dates, cases, marker='o', linestyle='-')
plt.title('Confirmed COVID-19 Cases in Hubei Province')
plt.xlabel('Date')
plt.ylabel('Number of Confirmed Cases')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


The graph illustrates the **progression of confirmed COVID-19 cases across various regions in Australia from March 1, 2020, to March 20, 2020.** Each line represents a specific region, showcasing the cumulative number of confirmed cases over the specified period.

Observing the graph, we can discern notable patterns and differences in the spread of the virus among the selected regions. Some regions exhibit a gradual increase in confirmed cases over time, while others experience more sporadic fluctuations.

Furthermore, certain regions may demonstrate a steeper incline in the number of cases, indicating a potentially rapid spread of the virus within those areas. Conversely, regions with a relatively flat line suggest slower rates of infection or effective containment measures.

[ ]
import matplotlib.pyplot as plt

# Data for selected regions
regions_data = {
    "Australian Capital Territory": [5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 14, 20, 25, 26, 26, 26, 24],
    "New South Wales": [0, 0, 0, 0, 0, 0, 0, 0, 2, 10, 12, 23, 33, 38, 42, 51, 55, 59, 64, 70],
    "Northern Territory": [1, 3, 5, 12, 12, 17, 17, 19, 20, 20, 20, 24, 26, 37, 48, 54, 60, 74, 87, 90],
    "Queensland": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 39, 39, 53, 75],
    "South Australia": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "Tasmania": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Victoria": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Western Australia": [0, 0, 1, 1, 1, 2, 8, 12, 12, 17, 19, 19, 31, 34, 45, 56, 68, 79, 97, 128]
}

# Plotting
plt.figure(figsize=(12, 8))
for region, data in regions_data.items():
    plt.plot(range(1, 21), data, marker='o', label=region)

plt.title('Confirmed COVID-19 Cases in Selected Australian Regions (3/1/20 - 3/20/20)')
plt.xlabel('Days (From 3/1/20 to 3/20/20)')
plt.ylabel('Number of Confirmed Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


A bar graph showing the cumulative number of confirmed COVID-19 cases for each region in Australia over the specified time period.
[ ]
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Date': ['3/1/20', '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20', '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20', '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20', '3/20/20'],
    'Australian Capital Territory': [5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 14, 20, 25, 26, 26, 26, 24],
    'New South Wales': [0, 0, 0, 0, 0, 0, 0, 0, 2, 10, 12, 23, 33, 38, 42, 51, 55, 59, 64, 70],
    'Northern Territory': [1, 3, 5, 12, 12, 17, 17, 19, 20, 20, 20, 24, 26, 37, 48, 54, 60, 74, 87, 90],
    'Queensland': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 18, 26, 52, 78, 84, 115, 136],
    'South Australia': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 6],
    'Tasmania': [6, 6, 13, 22, 22, 26, 28, 38, 48, 55, 65, 65, 92, 112, 134, 171, 210, 267, 307, 353],
    'Victoria': [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    'Western Australia': [9, 9, 11, 11, 13, 13, 13, 15, 15, 18, 20, 20, 35, 46, 61, 68, 78, 94, 144, 184]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 8))

# Loop through each region and plot the data
for region in df.columns[1:]:
    plt.bar(df['Date'], df[region], label=region)

plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('Confirmed COVID-19 Cases Over Time in Australian Regions')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


Correlation matrix
It is a table showing correlation coefficients between variables. It's a useful tool in statistics that helps understand the relationship between different variables in a dataset. Correlation coefficients measure the strength and direction of the linear relationship between two variables.

[ ]
import pandas as pd

# Data
data = {
    'Date': ['3/1/20', '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20', '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20', '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20', '3/20/20'],
    'Australian Capital Territory': [5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 14, 20, 25, 26, 26, 26, 24],
    'New South Wales': [0, 0, 0, 0, 0, 0, 0, 0, 2, 10, 12, 23, 33, 38, 42, 51, 55, 59, 64, 70],
    'Northern Territory': [1, 3, 5, 12, 12, 17, 17, 19, 20, 20, 20, 24, 26, 37, 48, 54, 60, 74, 87, 90],
    'Queensland': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 18, 26, 52, 78, 84, 115, 136],
    'South Australia': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 6],
    'Tasmania': [6, 6, 13, 22, 22, 26, 28, 38, 48, 55, 65, 65, 92, 112, 134, 171, 210, 267, 307, 353],
    'Victoria': [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    'Western Australia': [9, 9, 11, 11, 13, 13, 13, 15, 15, 18, 20, 20, 35, 46, 61, 68, 78, 94, 144, 184]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Compute correlation matrix
correlation_matrix = df.corr()

print("Correlation Matrix:")
print(correlation_matrix)

Correlation Matrix:
                              Australian Capital Territory  New South Wales  \
Australian Capital Territory                      1.000000         0.961517   
New South Wales                                   0.961517         1.000000   
Northern Territory                                0.947033         0.953984   
Queensland                                        0.880604         0.896918   
South Australia                                   0.829148         0.886754   
Tasmania                                          0.929127         0.953693   
Victoria                                          0.600041         0.719888   
Western Australia                                 0.856845         0.905303   

                              Northern Territory  Queensland  South Australia  \
Australian Capital Territory            0.947033    0.880604         0.829148   
New South Wales                         0.953984    0.896918         0.886754   
Northern Territory                      1.000000    0.955852         0.935066   
Queensland                              0.955852    1.000000         0.976496   
South Australia                         0.935066    0.976496         1.000000   
Tasmania                                0.990106    0.976931         0.965293   
Victoria                                0.697852    0.686959         0.754997   
Western Australia                       0.957703    0.979618         0.990511   

                              Tasmania  Victoria  Western Australia  
Australian Capital Territory  0.929127  0.600041           0.856845  
New South Wales               0.953693  0.719888           0.905303  
Northern Territory            0.990106  0.697852           0.957703  
Queensland                    0.976931  0.686959           0.979618  
South Australia               0.965293  0.754997           0.990511  
Tasmania                      1.000000  0.730571           0.976930  
Victoria                      0.730571  1.000000           0.756347  
Western Australia             0.976930  0.756347           1.000000  
<ipython-input-38-96252f1d2801>:20: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
  correlation_matrix = df.corr()
References
Pandas Documentation: (https://pandas.pydata.org/docs/).

Seaborn Documentation:(https://seaborn.pydata.org/tutorial.html).

Matplotlib Documentation:(https://matplotlib.org/stable/contents.html).

NumPy Documentation: (https://numpy.org/doc/).

Scikit-learn Documentation: (https://scikit-learn.org/stable/documentation.html).

Google Colab Documentation: (https://colab.research.google.com/notebooks/intro.ipynb).

Link to Recording https://drive.google.com/file/d/1oFedHvu9QQ42TJMTQZpSv6YFRFmBEGwO/view?usp=drive_link

