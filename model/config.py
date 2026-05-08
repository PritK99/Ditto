class Config:
    c_data_path = "data/c_tokens_with_lca_dist.parquet"   
    c_len = 110209
    cpp_data_path =  "data/cpp_tokens_with_lca_dist.parquet" 
    cpp_len = 48004
    vocab_path = "data/final_vocab_combined.txt"
    batch_size = 1
    max_seq_len = 1001    # This is including [SOS] or [EOS] (depending on encoder or decoder)
    use_lca_distance = True    # False is Baseline Relative Bias approach
    val_ratio = 0.05
    test_ratio = 0.05
    num_epochs = 2
    d_model = 512
    num_heads = 4
    vocab_size = 1001    # 997 + [UNK] + [SOS] + [EOS] + [PAD]
    max_pos = 273    # This is the max pos value taken by tokens
    pos_vocab_size = 547    # Positions go from [-max_pos, max_pos]. Hence, pos_vocab_size = 2*max_pos + 1. max_pos is 271. 
    num_encoders = 3
    num_decoders = 3
    ffn_hidden_size = 1024
    dropout = 0.1
    pad_idx = 0
    lr = 1e-4