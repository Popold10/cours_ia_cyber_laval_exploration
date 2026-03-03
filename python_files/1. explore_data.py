# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore the Midwest Survey dataset
#
# In this notebook, we will explore the **Midwest Survey** dataset from [skrub](https://skrub-data.org/).
#
# This dataset contains survey responses from people across the United States,
# asking them about their perception of the Midwest region.
#
# The goal is to predict the **Census Region** where a respondent lives,
# based on their survey answers.

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()

# X contains the features (the survey answers)
X = dataset.X
# y contains the target (the Census Region)
y = dataset.y

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?
#
# Use the `.shape` attribute to find out the number of rows and columns.

# %%
X.shape
(2494, 28)


 # %%
 X.head(10)
   RespondentID  ... In_what_ZIP_code_is_your_home_located
0  3.126807e+09  ...                                 74070
1  3.126791e+09  ...                                 44106
2  3.126781e+09  ...                                 48185
3  3.126770e+09  ...                                 45040
4  3.126765e+09  ...                                 44054
5  3.126757e+09  ...                                 61422
6  3.126746e+09  ...                                 63376
7  3.126738e+09  ...                                 60202
8  3.126736e+09  ...                                 16841
9  3.126729e+09  ...                                 73109

[10 rows x 28 columns]

# %% [markdown]
# ## Question 2: What is the distribution of the target?
#
# The target variable `y` tells us the Census Region of each respondent.
# Let's see how many respondents belong to each region.

# %%
y.value_counts()
Census_Region
East North Central    758
West North Central    358
Middle Atlantic       334
South Atlantic        248
Pacific               243
Mountain              190
West South Central    172
East South Central     97
New England            94
Name: count, dtype: int64


# %%
import matplotlib.pyplot as plt

counts = y.value_counts()

plt.figure(figsize=(8,6))
plt.barh(counts.index, counts.values)

plt.xlabel("Number of observations")
plt.ylabel("Census Region")
plt.title("Distribution of Census Regions")

plt.show()


# %% [markdown]
# Is the target balanced (roughly the same number of examples per class) or imbalanced?

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?
#
# Let's look at the column names and their data types.

# %%
list(X.columns)
['RespondentID', 'What_would_you_call_the_part_of_the_country_you_live_in_now', 'How_much_do_you_personally_identify_as_a_Midwesterner', 'Do_you_consider_Illinois_state_as_part_of_the_Midwest', 'Do_you_consider_Indiana_state_as_part_of_the_Midwest', 'Do_you_consider_Iowa_state_as_part_of_the_Midwest', 'Do_you_consider_Kansas_state_as_part_of_the_Midwest', 'Do_you_consider_Michigan_state_as_part_of_the_Midwest', 'Do_you_consider_Minnesota_state_as_part_of_the_Midwest', 'Do_you_consider_Missouri_state_as_part_of_the_Midwest', 'Do_you_consider_Nebraska_state_as_part_of_the_Midwest', 'Do_you_consider_North_Dakota_state_as_part_of_the_Midwest', 'Do_you_consider_Ohio_state_as_part_of_the_Midwest', 'Do_you_consider_South_Dakota_state_as_part_of_the_Midwest', 'Do_you_consider_Wisconsin_state_as_part_of_the_Midwest', 'Do_you_consider_Arkansas_state_as_part_of_the_Midwest', 'Do_you_consider_Colorado_state_as_part_of_the_Midwest', 'Do_you_consider_Kentucky_state_as_part_of_the_Midwest', 'Do_you_consider_Oklahoma_state_as_part_of_the_Midwest', 'Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest', 'Do_you_consider_West_Virginia_state_as_part_of_the_Midwest', 'Do_you_consider_Montana_state_as_part_of_the_Midwest', 'Do_you_consider_Wyoming_state_as_part_of_the_Midwest', 'Gender', 'Age', 'Household_Income', 'Education', 'In_what_ZIP_code_is_your_home_located']


 # %%
 print(X.dtypes)
RespondentID                                                   float64
What_would_you_call_the_part_of_the_country_you_live_in_now        str
How_much_do_you_personally_identify_as_a_Midwesterner              str
Do_you_consider_Illinois_state_as_part_of_the_Midwest              str
Do_you_consider_Indiana_state_as_part_of_the_Midwest               str
Do_you_consider_Iowa_state_as_part_of_the_Midwest                  str
Do_you_consider_Kansas_state_as_part_of_the_Midwest                str
Do_you_consider_Michigan_state_as_part_of_the_Midwest              str
Do_you_consider_Minnesota_state_as_part_of_the_Midwest             str
Do_you_consider_Missouri_state_as_part_of_the_Midwest              str
Do_you_consider_Nebraska_state_as_part_of_the_Midwest              str
Do_you_consider_North_Dakota_state_as_part_of_the_Midwest          str
Do_you_consider_Ohio_state_as_part_of_the_Midwest                  str
Do_you_consider_South_Dakota_state_as_part_of_the_Midwest          str
Do_you_consider_Wisconsin_state_as_part_of_the_Midwest             str
Do_you_consider_Arkansas_state_as_part_of_the_Midwest              str
Do_you_consider_Colorado_state_as_part_of_the_Midwest              str
Do_you_consider_Kentucky_state_as_part_of_the_Midwest              str
Do_you_consider_Oklahoma_state_as_part_of_the_Midwest              str
Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest          str
Do_you_consider_West_Virginia_state_as_part_of_the_Midwest         str
Do_you_consider_Montana_state_as_part_of_the_Midwest               str
Do_you_consider_Wyoming_state_as_part_of_the_Midwest               str
Gender                                                             str
Age                                                                str
Household_Income                                                   str
Education                                                          str
In_what_ZIP_code_is_your_home_located                              str
dtype: object


# %% [markdown]
# How many features are numerical? How many are categorical (text)?

# %%

# %%
from skrub import TableReport
... report=TableReport(X)
... report.open()

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?
#
# Missing values can cause problems for machine learning models.
# Let's check if there are any.

# %%
X.isna().sum()
RespondentID                                                   0
What_would_you_call_the_part_of_the_country_you_live_in_now    0
How_much_do_you_personally_identify_as_a_Midwesterner          0
Do_you_consider_Illinois_state_as_part_of_the_Midwest          0
Do_you_consider_Indiana_state_as_part_of_the_Midwest           0
Do_you_consider_Iowa_state_as_part_of_the_Midwest              0
Do_you_consider_Kansas_state_as_part_of_the_Midwest            0
Do_you_consider_Michigan_state_as_part_of_the_Midwest          0
Do_you_consider_Minnesota_state_as_part_of_the_Midwest         0
Do_you_consider_Missouri_state_as_part_of_the_Midwest          0
Do_you_consider_Nebraska_state_as_part_of_the_Midwest          0
Do_you_consider_North_Dakota_state_as_part_of_the_Midwest      0
Do_you_consider_Ohio_state_as_part_of_the_Midwest              0
Do_you_consider_South_Dakota_state_as_part_of_the_Midwest      0
Do_you_consider_Wisconsin_state_as_part_of_the_Midwest         0
Do_you_consider_Arkansas_state_as_part_of_the_Midwest          0
Do_you_consider_Colorado_state_as_part_of_the_Midwest          0
Do_you_consider_Kentucky_state_as_part_of_the_Midwest          0
Do_you_consider_Oklahoma_state_as_part_of_the_Midwest          0
Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest      0
Do_you_consider_West_Virginia_state_as_part_of_the_Midwest     0
Do_you_consider_Montana_state_as_part_of_the_Midwest           0
Do_you_consider_Wyoming_state_as_part_of_the_Midwest           0
Gender                                                         0
Age                                                            0
Household_Income                                               0
Education                                                      0
In_what_ZIP_code_is_your_home_located                          0
dtype: int64


# %% [markdown]
# Missing values can sometimes be encoded differently. Let's look at some columns more closely.

 # %%
 X["Household_Income"].unique()
<StringArray>
[  '$50,000 - $99,999',        '$0 - $24,999',   '$25,000 - $49,999',
           '$150,000+', '$100,000 - $149,999',                   '?']
Length: 6, dtype: str

# %%
X["Education"].unique()
<StringArray>
[          'High school degree', 'Associate or bachelor degree',
              'Graduate degree',                 'Some college',
 'Less than high school degree',                            '?']
Length: 6, dtype: str

# %% [markdown]
# Do you see a special value that could represent missing data?

# %% [markdown]
# ## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?
#
# Let's explore this important feature.

# %%
counts() = X["How_much_do_you_personally_identify_as_a_Midwesterner"].val\
... print(counts)
...
How_much_do_you_personally_identify_as_a_Midwesterner
Not at all    965
A lot         697
Some          528
Not much      304
Name: count, dtype: int64


# %%
import matplotlib.pyplot as plt

counts = X["How_much_do_you_personally_identify_as_a_Midwesterner"].value_counts()

plt.figure(figsize=(8,6))
counts.plot(kind="barh", color="skyblue")  
plt.xlabel("Number of respondents")
plt.ylabel("Level of Midwestern Identification")
plt.title("Distribution of Midwestern Identification")
plt.gca().invert_yaxis()  
plt.show()


# %% [markdown]
# ## Bonus: Explore another feature
#
# Pick another column and explore its distribution.
# For example: `Gender`, `Age`, or one of the
# "Do you consider X state as part of the Midwest" columns.

# %%
import matplotlib.pyplot as plt
...
...
... print("Unique values:")
... print(X["Household_Income"].unique())
...
...
... counts = X["Household_Income"].value_counts()
... print("\nValue counts:")
... print(counts)
...
...
... plt.figure(figsize=(8,6))
... counts.plot(kind="barh", color="lightgreen")
... plt.xlabel("Number of respondents")
... plt.ylabel("Household Income")
... plt.title("Distribution of Household Income")
... plt.gca().invert_yaxis()  # barre la plus grande en haut
... plt.show()
...
Unique values:
<StringArray>
[  '$50,000 - $99,999',        '$0 - $24,999',   '$25,000 - $49,999',
           '$150,000+', '$100,000 - $149,999',                   '?']
Length: 6, dtype: str

Value counts:
Household_Income
$50,000 - $99,999      720
$150,000+              682
$25,000 - $49,999      421
$100,000 - $149,999    361
$0 - $24,999           242
?                       68
Name: count, dtype: int64

