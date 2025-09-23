import pandas as pd
import numpy as np

# Load the dataset from GitHub URL
url = 'https://github.com/melaniewalsh/Intro-Cultural-Analytics/raw/master/book/data/bellevue_almshouse_modified.csv'
df_bellevue = pd.read_csv(url)


# Exercise 1: Fibonacci function
def fib(n):
    """
    Calculate the nth number in the Fibonacci series using recursion.
    The series starts with 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...
    
    Args:
        n: The position in the Fibonacci series (0-indexed)
    
    Returns:
        The nth Fibonacci number
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    return fib(n - 1) + fib(n - 2)



fibonacci = fib


# Exercise 2: Binary conversion
def to_binary(n):
    """
    Convert an integer to its binary representation using recursion.
    
    Args:
        n: Integer to convert to binary
    
    Returns:
        Binary representation as a string (e.g., 12 returns '1100')
    """
    if n == 0:
        return '0'
    elif n == 1:
        return '1'
    
    return to_binary(n // 2) + str(n % 2)


# Exercise 3: Data analysis tasks
def task_1():
    """
    Sort columns by missing values (least to most).
    Returns a list of column names.
    """
    global df_bellevue
    
    if 'gender' in df_bellevue.columns:
        print("Fixing gender column issues...")
        df_bellevue['gender'] = df_bellevue['gender'].astype(str)
        df_bellevue['gender'] = df_bellevue['gender'].str.strip()
        df_bellevue.loc[df_bellevue['gender'] == 'nan', 'gender'] = np.nan
    
    missing_counts = df_bellevue.isnull().sum()
    
    sorted_columns = missing_counts.sort_values().index.tolist()
    
    print(f"Columns sorted from least ({missing_counts.min()}) to most ({missing_counts.max()}) missing values")
    
    return sorted_columns


def task_2():
    """
    Year and admissions dataframe.
    Returns a DataFrame with 'year' and 'total_admissions' columns.
    """
    global df_bellevue
    
    df_temp = df_bellevue.copy()
    
    year_extracted = False
    
    if 'year' in df_temp.columns:
        year_col = 'year'
        year_extracted = True
    elif 'date_in' in df_temp.columns:
        print("Extracting year from 'date_in' column...")
        # Convert to datetime and extract year
        df_temp['year'] = pd.to_datetime(df_temp['date_in'], errors='coerce').dt.year
        year_col = 'year'
        year_extracted = True
    else:
        for col in df_temp.columns:
            if 'date' in col.lower() or 'year' in col.lower():
                print(f"Found date/year column: {col}")
                if 'date' in col.lower():
                    df_temp['year'] = pd.to_datetime(df_temp[col], errors='coerce').dt.year
                else:
                    df_temp['year'] = pd.to_numeric(df_temp[col], errors='coerce')
                year_col = 'year'
                year_extracted = True
                break
    
    if not year_extracted:
        print("Warning: No year or date column found!")
        return pd.DataFrame({'year': [], 'total_admissions': []})
    
    df_temp = df_temp[df_temp[year_col].notna()]
    
    result = df_temp.groupby(year_col).size().reset_index(name='total_admissions')
    result.columns = ['year', 'total_admissions']
    
    result['year'] = result['year'].astype(int)
    
    print(f"Found data for {len(result)} years, from {result['year'].min()} to {result['year'].max()}")
    
    return result


def task_3():
    """
    Average age by gender series.
    Returns a Series with gender as index and average age as values.
    """
    global df_bellevue
    
    # Make a copy to avoid modifying the original
    df_temp = df_bellevue.copy()
    
    # Clean gender column
    if 'gender' in df_temp.columns:
        df_temp['gender'] = df_temp['gender'].astype(str).str.strip()
        # Remove 'nan' strings
        df_temp.loc[df_temp['gender'] == 'nan', 'gender'] = np.nan
        # Remove empty strings
        df_temp.loc[df_temp['gender'] == '', 'gender'] = np.nan
    else:
        print("Warning: No 'gender' column found!")
        return pd.Series()
    
    # Find age column
    age_col = None
    for col in df_temp.columns:
        if 'age' in col.lower():
            age_col = col
            print(f"Using column '{col}' for age data")
            break
    
    if age_col is None:
        print("Warning: No age column found!")
        return pd.Series()
    
    df_temp[age_col] = pd.to_numeric(df_temp[age_col], errors='coerce')
    
    df_clean = df_temp.dropna(subset=['gender', age_col])
    
    result = df_clean.groupby('gender')[age_col].mean()
    
    print(f"Found {len(result)} gender categories")
    print(f"Processed {len(df_clean)} records with valid gender and age data")
    
    return result


def task_4():
    """
    Top 5 professions list.
    Returns a list of the 5 most common professions in order of prevalence.
    """
    global df_bellevue
    
    df_temp = df_bellevue.copy()
    
    prof_col = None
    for col in df_temp.columns:
        if any(term in col.lower() for term in ['profession', 'occupation', 'trade', 'job', 'work']):
            prof_col = col
            print(f"Using column '{col}' for profession data")
            break
    
    if prof_col is None:
        print("Warning: No profession/occupation column found!")
        print("Available columns:", df_temp.columns.tolist())
        return []
    
    df_temp[prof_col] = df_temp[prof_col].astype(str).str.strip()
    
    valid_professions = df_temp[prof_col][
        (df_temp[prof_col] != 'nan') & 
        (df_temp[prof_col] != '') &
        (df_temp[prof_col].notna())
    ]
    
    profession_counts = valid_professions.value_counts()
    
    if len(profession_counts) == 0:
        print("Warning: No valid professions found!")
        return []
    
    top_5_professions = profession_counts.head(5).index.tolist()
    
    print(f"Total unique professions: {len(profession_counts)}")
    print(f"Top 5 professions account for {profession_counts.head(5).sum()} out of {len(valid_professions)} entries")
    print(f"Most common: {top_5_professions[0]} with {profession_counts.iloc[0]} occurrences")
    
    return top_5_professions


# Testing section (optional)
if __name__ == "__main__":
    

    # Test Exercise 1: Fibonacci
    print("\n=== Exercise 1: Fibonacci Series ===")
    test_cases = [0, 1, 2, 5, 9, 10]
    for n in test_cases:
        print(f"fib({n}) = {fib(n)}")
    
    # Test Exercise 2: Binary Conversion
    print("\n=== Exercise 2: Binary Conversion ===")
    test_numbers = [0, 1, 2, 7, 12, 15, 255]
    for num in test_numbers:
        binary = to_binary(num)
        expected = bin(num)[2:]
        check = "✓" if binary == expected else "✗"
        print(f"to_binary({num}) = {binary} {check}")
    
    # Test Exercise 3: Data Analysis
    print("\n=== Exercise 3: Data Analysis Tasks ===")
    
    try:
        print("\nDataset info:")
        print(f"Shape: {df_bellevue.shape}")
        print(f"Columns: {df_bellevue.columns.tolist()}")
        
        print("\n--- Task 1: Columns by missing values ---")
        result1 = task_1()
        print(f"First 3 columns (least missing): {result1[:3]}")
        print(f"Last 3 columns (most missing): {result1[-3:]}")
        
        print("\n--- Task 2: Admissions by year ---")
        result2 = task_2()
        if not result2.empty:
            print(result2.head())
            print(f"Total years: {len(result2)}")
            print(f"Total admissions: {result2['total_admissions'].sum()}")
        
        print("\n--- Task 3: Average age by gender ---")
        result3 = task_3()
        if not result3.empty:
            print(result3)
            print(f"Average age across all genders: {result3.mean():.2f}")
        
        print("\n--- Task 4: Top 5 professions ---")
        result4 = task_4()
        if result4:
            for i, prof in enumerate(result4, 1):
                print(f"{i}. {prof}")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please check your internet connection and that the URL is accessible.")
        print("URL:", url)