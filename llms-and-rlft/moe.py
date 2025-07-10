from dd_tokenizer_base_v1 import SMILESBPETokenizer
from dd_tokenizer_base_v1 import LDataModule
filename = "Your_Data_Path"
checkpoint = "Your_Check_Path"
hyperparams = {"batch_size": 60, "max_epochs": 1, "min_epochs": 1,
               "max_length": 64, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_076, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 8, "n_embd": 12 * 48}
num_workers = 4  # Number of dataloader worker processes.
tokenizer = SMILESBPETokenizer(dropout=None)
tokenizer = SMILESBPETokenizer.get_hf_tokenizer('checkpoints/250k/tokenizer.json', model_max_length=hyperparams["max_length"])
datamodule = LDataModule(filename, tokenizer,
                              batch_size=hyperparams["batch_size"],
                              num_workers=num_workers)
from transformers import OlmoeConfig, OlmoeForCausalLM
print(tokenizer.vocab_size)
config    =  OlmoeConfig(vocab_size = tokenizer.vocab_size,
                         hidden_size=192,#num-heads*N
                         intermediate_size = 1024,
                         max_position_embeddings = 128,
                         num_hidden_layer = 12,
                         num_attention_heads = 16,
                         hidden_act = 'gelu',
                         find_unused_parameters= False,
                         #rotary_pct = 0.25,
                         #rotary_emb_base = 10000,
                         attention_dropout = 0.0,
                         #hidden_dropout = 0.0,
                         initializer_range = 0.02,
                         #layer_norm_eps = 1e-05,
                         use_cache = True,
                         bos_token_id = tokenizer.bos_token_id,
                         eos_token_id = tokenizer.eos_token_id,
                         tie_word_embeddings = False,
                         #use_parallel_residual = True,
                         rope_scaling = None,
                         attention_bias = True,
                         output_router_logits=True,
                        )
model = OlmoeForCausalLM(config)
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
print(tokenizer.vocab_size)
config =  Qwen3MoeConfig(vocab_size = tokenizer.vocab_size,
                         hidden_size=96,
                         intermediate_size = 1024,
                         max_position_embeddings = 64,
                         num_hidden_layer = 12, #0.75*12
                         num_attention_heads = 16,
                         hidden_act = 'silu',
                         attention_dropout = 0.0,
                         initializer_range = 0.02,
                         use_cache = True,
                         bos_token_id = tokenizer.bos_token_id,
                         eos_token_id = tokenizer.eos_token_id,
                         tie_word_embeddings = False,
                         rope_scaling = None,
                         attention_bias = True,
                         output_router_logits=True,
                        )
model = Qwen3MoeForCausalLM(config)
from transformers import DeepseekV3Config,DeepseekV3ForCausalLM
print(tokenizer.vocab_size)
config =DeepseekV3Config(vocab_size = tokenizer.vocab_size,
                         hidden_size= 72,
                         intermediate_size = 128,
                         max_position_embeddings = 32,
                         moe_intermediate_size = 256,
                         n_shared_experts = 1,
                         n_routed_experts = 128,
                         num_hidden_layer = 12, #0.75*12
                         num_attention_heads = 12,
                         num_key_value_heads=12,
                         hidden_act = 'silu',
                         #rotary_pct = 0.25,
                         #rotary_emb_base = 10000,
                         attention_dropout = 0.0,
                         #hidden_dropout = 0.0,
                         initializer_range = 0.02,
                         #layer_norm_eps = 1e-05,
                         use_cache = True,
                         bos_token_id = tokenizer.bos_token_id,
                         eos_token_id = tokenizer.eos_token_id,
                         tie_word_embeddings = False,
                         #use_parallel_residual = True,
                         rope_scaling = None,
                         attention_bias = True,
                         output_router_logits=True,
                        )
model = DeepseekV3ForCausalLM(config)
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
config =GraniteMoeConfig(vocab_size = tokenizer.vocab_size,
                         hidden_size= 256,
                         intermediate_size = 256,
                         max_position_embeddings = 32,
                         moe_intermediate_size = 256,
                         num_local_experts = 128,
                         num_hidden_layer = 12, #0.75*12
                         num_attention_heads = 16,
                         num_key_value_heads=16,
                         hidden_act = 'silu',
                         attention_dropout = 0.0,
                         initializer_range = 0.02,
                         use_cache = True,
                         bos_token_id = tokenizer.bos_token_id,
                         eos_token_id = tokenizer.eos_token_id,
                         tie_word_embeddings = False,
                         rope_scaling = None,
                         attention_bias = True,
                         output_router_logits=True,
                        )
model = GraniteMoeForCausalLM(config)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
checkpoint_cb = ModelCheckpoint(f"{checkpoint}/model/")
early_stopping_ppl = EarlyStopping(
    monitor="ppl_epoch",
    patience=2,
    min_delta=5e-3,
    check_finite=True,
    stopping_threshold=1.1,
    divergence_threshold=hyperparams["vocab_size"] / 10,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=True,
)
trainer = Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=-1,
    callbacks=[checkpoint_cb, early_stopping_ppl],
    max_epochs=hyperparams["max_epochs"],
    min_epochs=hyperparams["min_epochs"],
    val_check_interval=0.4,
    limit_train_batches=0.5,
    log_every_n_steps=200,
)
lit_model = llm.LLMLitModel(
    model,
    batch_size=hyperparams["batch_size"],
    learning_rate=hyperparams["learning_rate"],
    final_learning_rate=hyperparams["final_learning_rate"],
    weight_decay=hyperparams["weight_decay"],
    adam_eps=hyperparams["adam_eps"],
    adam_betas=hyperparams["adam_betas"],
    scheduler_T_max=hyperparams["scheduler_T_max"],
)
trainer.fit(lit_model, datamodule)
lit_model.transformer.save_pretrained(f"{checkpoint}/model/")
#nohup python 1-gemma-lora-ft.py > 1_run.log 2>&1 &
