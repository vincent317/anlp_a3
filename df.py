import pandas as pd

# Given list of numbers
numbers = [
    10.187240600585938, 
    19.774027585983276, 
    40.74758529663086, 
    62.89952063560486, 
    80.89390206336975, 
    101.32650256156921
]

# Convert list to a DataFrame row
df = pd.DataFrame([numbers], columns=[f'Value_{i+1}' for i in range(len(numbers))])

# Display the DataFrame
print(df)
