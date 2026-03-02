import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_dataset_summary(train_path, test_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 1. Basic Dimensions
    print("--- Basic Dimensions ---")
    print(f"Training set size: {len(train_df)} rows")
    print(f"Testing set size:  {len(test_df)} rows")
    
    # 2. Class Balance (Label Distribution)
    print("\n--- Class Balance (Label Counts) ---")
    train_counts = train_df['label'].value_counts()
    test_counts = test_df['label'].value_counts()
    
    balance_df = pd.DataFrame({
        'Train Count': train_counts,
        'Train %': (train_counts / len(train_df) * 100).round(2),
        'Test Count': test_counts,
        'Test %': (test_counts / len(test_df) * 100).round(2)
    })
    print(balance_df)
    
    # 3. Tweet Length Analysis
    train_df['char_length'] = train_df['tweet'].astype(str).apply(len)
    train_df['word_count'] = train_df['tweet'].astype(str).apply(lambda x: len(x.split()))
    
    print("\n--- Tweet Statistics (Training Set) ---")
    print(f"Average Character Length: {train_df['char_length'].mean():.2f}")
    print(f"Average Word Count:       {train_df['word_count'].mean():.2f}")
    print(f"Max Word Count:           {train_df['word_count'].max()}")

    # 4. Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Class Balance
    sns.countplot(x='label', data=train_df, ax=ax1, palette='viridis')
    ax1.set_title('Class Distribution in Training Set')
    ax1.set_xlabel('Label (0: Non-Personal, 1: Personal)')
    
    # Plot Word Count Distribution
    sns.histplot(train_df['word_count'], bins=30, ax=ax2, color='skyblue', kde=True)
    ax2.set_title('Distribution of Word Counts')
    ax2.set_xlabel('Number of Words per Tweet')
    
    plt.tight_layout()
    plt.show()

# Execute summary
get_dataset_summary('phm_train.csv', 'phm_test.csv')