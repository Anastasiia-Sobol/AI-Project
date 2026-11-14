# Importing Relevant Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the CSV File into an initial Dataframe
df_property_rental=pd.read_csv("Irish_Rent_Prices_2020_2025.csv")
print(df_property_rental.head())
print(df_property_rental.columns)

print(df_property_rental.shape)
print(df_property_rental.describe())
print("\nInformation about the Dataset:")
print(df_property_rental.info())
print("\nChecking for duplicate rows:")
print(df_property_rental.duplicated().sum())


# Analysis of Missing Values, which are all found in the VALUE Column
print("\nDisplaying the number of blank values, i.e. NaN values present:")
print(df_property_rental.isna().sum())
missingValues=df_property_rental['VALUE'].isna().sum()
availableValues=df_property_rental['VALUE'].notna().sum()

plt.pie([availableValues, missingValues], labels=['Available','Missing'], autopct='%1.1f%%')
plt.title("Comparison of Missing and Available Values in the VALUE column")
plt.show()

# Statistical analysis of the VALUE column
print("\nStatistical Information about the VALUE column:")
print(df_property_rental['VALUE'].describe())

# Shows how many categories belong to the Number of Bedrooms Column
print("\nDisplaying the frequency of each value in a column:")
print(df_property_rental['Number of Bedrooms'].value_counts())



# Analysis of the distribution of missing values in the VALUE column across other categories
print("\nInvestigating whether the missing values in the VALUE column are evenly distributed by Number of Rooms")
print(df_property_rental['VALUE'].isna().groupby(df_property_rental['Number of Bedrooms']).value_counts())

print("\nInvestigating whether the missing values in the VALUE column are evenly distributed by Location")
print(df_property_rental['VALUE'].isna().groupby(df_property_rental['Location']).value_counts())

print("\nInvestigating whether the missing values in the VALUE column are evenly distributed by Property Type")
print(df_property_rental['VALUE'].isna().groupby(df_property_rental['Property Type']).value_counts())


print("\nInvestigating whether the missing values in the VALUE column are evenly distributed by Quarter")
print(df_property_rental['VALUE'].isna().groupby(df_property_rental['Quarter']).value_counts())

missingbyQuarter=df_property_rental['VALUE'].isna().groupby(df_property_rental['Quarter']).sum()
availablebyQuarter=df_property_rental['VALUE'].notna().groupby(df_property_rental['Quarter']).sum()

print(missingbyQuarter)
print(availablebyQuarter)

df_rental_quarterly=pd.concat([missingbyQuarter, availablebyQuarter], axis=1)
df_rental_quarterly.columns=["Missing Values", "Available Values"]
print(df_rental_quarterly)
print(df_rental_quarterly.shape)

print(df_rental_quarterly.index)
plt.bar(df_rental_quarterly.index, df_rental_quarterly['Missing Values'], label='Missing Values')
plt.bar(
    df_rental_quarterly.index,
    df_rental_quarterly['Available Values'],
    bottom=df_rental_quarterly['Missing Values'],
    label='Available Values'
)
plt.legend()
plt.title('Missing and Available Rental Values by Quarter')
plt.xticks(rotation=90)
plt.show()



# Creating an analysing a new dataframe with missing values dropped
df_property_rental_full=df_property_rental.dropna(subset=['VALUE'])

print("\nFollowing creation of the dataframe with missing values eliminated, this reveals the structure of the new dataframe")
print(df_property_rental_full.shape)
print(df_property_rental_full['VALUE'].isna().sum())
print(df_property_rental.shape)


# Examination of availabilility of data for the most up-to-date quarter, 2025Q4, by All Property Types
print("\nInvestigating the availability of the most recent data, 2025 Q4, by All Property Types")
print(
    df_property_rental[
        (df_property_rental['Quarter'] == '2025Q4') &
        (df_property_rental['Property Type'] == 'All property types')
    ]['VALUE'].isna().value_counts()
)

# Filtering this data
df_rental_2025q4=df_property_rental_full[
    (df_property_rental_full['Quarter']=='2025Q4') &
    (df_property_rental_full['Property Type'] == 'All property types')
]
print(df_rental_2025q4.shape)


# Creating a new filtered version of df_rental_2025q4 that will explore 2025 Q4 Rental Data by Location
print("\nInvestigation of 2025 Q4 Rental Data by Location")
df_rental_location_2025q4=df_rental_2025q4[
    (df_rental_2025q4['Number of Bedrooms'] == 'All bedrooms') &
    (df_rental_2025q4['Property Type'] == 'All property types')
]
print(df_rental_location_2025q4.shape)

rent_by_location=df_rental_location_2025q4.groupby('Location')['VALUE'].mean()
print(rent_by_location)

print(f"Highest average monthly rent, found in {rent_by_location.idxmax()} was {rent_by_location.max()}")
print(f"Lowest average monthly rent, found in {rent_by_location.idxmin()} was {rent_by_location.min()}")

top10=rent_by_location.nlargest(10)
bottom10=rent_by_location.nsmallest(10)

plt.bar(top10.index, top10)
plt.xlabel("Location")
plt.ylabel("Average Monthly Rent (€)")
plt.title("10 Most Expensive Rental Locations in Ireland (2025 Q4)")
plt.xticks(rotation=15)
plt.show()

plt.bar(bottom10.index, bottom10)
plt.xlabel("Location")
plt.ylabel("Average Monthly Rent (€)")
plt.title("10 Least Expensive Rental Locations in Ireland (2025 Q4)")
plt.xticks(rotation=15)
plt.show()



# Analysing how rent has changed over time, filtering VALUE on All Bedrooms and All Property Types    
df_rental_over_time=df_property_rental_full[
    (df_property_rental_full['Number of Bedrooms'] == 'All bedrooms') &
    (df_property_rental_full['Property Type'] == 'All property types')
]
print(df_rental_over_time.shape)
quarterly_rent = df_rental_over_time.groupby('Quarter')['VALUE'].mean()
print(quarterly_rent)

# Uses the reset_index function on the Quarterly Rent variable, as Quarter is the index of this variable
df_quarterly_rent = quarterly_rent.reset_index()
print(df_quarterly_rent.head())

df_quarterly_rent['Quarter Number']=list(range(1,25))
print(df_quarterly_rent)


print("\nCalculates the correlation between time and rent amount (filtered by All Bedroom types and All Property types)")
print(df_quarterly_rent.corr(numeric_only=True)["VALUE"].abs().sort_values(ascending=False))

# Displays this on a scatter plot
plt.scatter(df_quarterly_rent["Quarter Number"], df_quarterly_rent["VALUE"])
plt.xlabel("Quarter")
plt.ylabel("Average Monthly Rent (€)")
plt.title("Correlation Between Time and Average Rent")
plt.show()

# Displays this on a line plot with markers
plt.plot(df_quarterly_rent["Quarter Number"], df_quarterly_rent["VALUE"], marker="o")
plt.xlabel("Quarter")
plt.ylabel("Average Monthly Rent (€)")
plt.title("Average Monthly Rent Over Time")
plt.show()