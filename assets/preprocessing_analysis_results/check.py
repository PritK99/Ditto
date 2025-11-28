import pandas as pd

# Load both vocab files
df1 = pd.read_csv("c/vocab_counts.csv")
df2 = pd.read_csv("cpp/vocab_counts.csv")   

combined_df = pd.concat([df1, df2], axis=0)

combined_df = combined_df.groupby('token', as_index=False).sum()

print("Combined vocab size:", len(combined_df))

num_rows_filters = [10]

for num_rows_filter in num_rows_filters:
    filtered_df = combined_df[
        (combined_df['num_rows'] > num_rows_filter) 
    ]

    print(f"Rows after filtering num_rows>{num_rows_filter}:",
        len(filtered_df))
