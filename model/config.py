class Config:
    c_data_path = "../data/c_tokens_with_lca_dist.parquet"   
    c_len = 110209
    cpp_data_path =  "../data/cpp_tokens_with_lca_dist.parquet" 
    cpp_len = 48004
    vocab_path = "../data/final_vocab_combined.txt"
    max_seq_len = 1000    # This is excluding [SOS] and [EOS]
    use_lca_distance = True    # False is Baseline Relative Bias approach
    val_ratio = 0.05
    test_ratio = 0.05