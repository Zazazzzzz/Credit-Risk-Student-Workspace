## Shazia
# choosing Germany and France for the comparison

#### Step 1 ####

# All needed libraries
# pip install pandas matplotlib openpyxl
import pandas as pd
import matplotlib.pyplot as plt
# importing data set
df = pd.read_excel('real_GDP_growth.xlsx')
# print(df.dtypes)
# now changing first column name to short it
df = df.rename(columns={df.columns[0]: 'country'})
# reshaping data long to wide using pd.melt()
df = pd.melt(df, id_vars='country', var_name='year', value_name='gdp_growth')
# data types conversion
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['gdp_growth'] = pd.to_numeric(df['gdp_growth'], errors='coerce')
# removing na values from the data to clean the data
df = df.replace('no data', pd.NA)
# 7. sort and filter
df = df.sort_values(['country', 'year']).reset_index(drop=True)
df = df[df['country'] != 'West Bank and Gaza']
print(df.head(10))
print(df.dtypes)

###### Step 2 ######
df['avg_growth'] = df.groupby('country')['gdp_growth'].transform('mean')
print(df[df['country'] == 'Germany'].head(5))

###### step 3 ####
two_countries = df[df['country'].isin(['Germany', 'France'])]

# reshaping data to wide using pivot
plot_data = two_countries.pivot(index='year', columns='country', values='gdp_growth')

plt.figure(figsize=(12, 6))

plt.plot(plot_data.index, plot_data['Germany'], label='Germany', color='black')
plt.plot(plot_data.index, plot_data['France'], label='France', color='blue')

plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)  # zero reference line

plt.title('Real GDP Growth: Germany vs France')
plt.xlabel('Year')
plt.ylabel('GDP Growth (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()