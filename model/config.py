class Config:
    c_data_path = "../data/c_cleaned_tokens.csv"   
    cpp_data_path =  "../data/cpp_cleaned_tokens.csv" 
    vocab_path = "../data/final_vocab_combined.txt"
    max_seq_len = 1000    # This is excluding [SOS] and [EOS]
    use_lca_distance = False    # Baseline approach
    val_ratio = 0.05
    test_ratio = 0.05