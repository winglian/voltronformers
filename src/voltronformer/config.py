from src.voltronformer.utils import DictDefault


def teeny():
    """50M parameters"""
    return DictDefault({
        # "tokenizer_name": "mistralai/Mistral-Nemo-Instruct-2407",
        "tokenizer_name": "mistralai/Mistral-7B-v0.1",
    }), DictDefault({
        "hidden_size": 512,
        "intermediate_size": 1408,
        "rope_theta": 10_000,
        "max_position_embeddings": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_hidden_layers": 12,
        "vocab_size": 32768,
        "dwa_dilation": 4,
        "dwa_period": 5,
        "pad_token_id": 2,
        "mod_every": 2,
        "mod_capacity_factor": 0.125,
        "rms_norm_eps": 0.000001,
        "dwa": True,
        "infini_attention": True,
        "ia_segment_length": 512,
        "ia_dim_head": 64,
        "ia_delta_rule": True,
        "dropout": 0.1,
    })


def tiny():
    """300M parameters"""
    return DictDefault({

    }), DictDefault({
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "rope_theta": 10_000,
        "max_position_embeddings": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 24,
        "vocab_size": 131072,
        "dwa_dilation": 4,
        "dwa_period": 5,
        "pad_token_id": 2,
        "mod_every": 2,
        "mod_capacity_factor": 0.125,
        "rms_norm_eps": 0.000001,
        "dwa": True,
        "infini_attention": True,
        "ia_segment_length": 512,
        "ia_dim_head": 64,
        "ia_delta_rule": True,
        "dropout": 0.1,
    })


def small():
    """1.1B parameters"""
    return DictDefault({

    }), DictDefault({
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "rope_theta": 10_000,
        "max_position_embeddings": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 24,
        "vocab_size": 131072,
        "dwa_dilation": 4,
        "dwa_period": 5,
        "pad_token_id": 2,
        "mod_every": 2,
        "mod_capacity_factor": 0.125,
        "rms_norm_eps": 0.000001,
        "dwa": True,
        "infini_attention": True,
        "ia_segment_length": 512,
        "ia_dim_head": 128,
        "ia_delta_rule": True,
        "dropout": 0.1,
    })
