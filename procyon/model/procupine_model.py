#this is the wrapper for our encoder, analogous to the ESM PLM
#commented out things I wasnt sure wed need like pooling, splitting sequences, etc
from procupine_encoder import procupineVAE
class Procupine_PLM(torch.nn.Module):
    def __init__(
        self,
        pretrained_weights_dir,
        num_params = '3b',
        pooling_method = 'max',
        padding_idx = 1,
        # eos_idx = 2,
        max_protein_len = 1024,
        # long_rna_strategy = 'split',
        max_batch_forward_pass = None,
        use_lora = False,
        use_q_lora = False,
        use_task_spc_lora = False,
        lora_alpha = 8,
        lora_r = 8,
        use_adapter = False,
        adapter_rank = 8,
        use_prefix=False,
        prefix_dropout=0.0,
        prefix_mid_dim=800,
        prefix_attn_bn=30,
        protein_attention_type = 'vanilla',
        lora_parameters = 'default',
        lora_num = 2,
    ):
        super(Procupine_PLM, self).__init__()

        self.vae = procupineVAE(gene_dim, hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        self.use_lora = use_lora

        self.seq_proc = partial(batched_split_long_seq,
            padding_idx = self.padding_idx,
            eos_idx = self.eos_idx,
            long_protein_strategy = self.long_protein_strategy,
            max_protein_len = self.max_protein_len)
        if self.num_params == '8m':
            self.model, _ = esm.pretrained.esm2_t6_8M_UR50D()
            #self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t6_8M_UR50D.pt')
            self.repr_layer = 6
            self.embedding_size = 320
        elif self.num_params == '35m':
            self.model, _ = esm.pretrained.esm2_t12_35M_UR50D()
            #self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t12_35M_UR50D.pt')
            self.repr_layer = 12
            self.embedding_size = 480
        elif self.num_params == '650m':
            self.model, _ = esm.pretrained.esm2_t33_650M_UR50D()
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t33_650M_UR50D.pt')
            self.repr_layer = 33
            self.embedding_size = 1280
        elif self.num_params == '3b':
            self.model, _ = esm.pretrained.esm2_t36_3B_UR50D()
            # FIXME: This local loading is not working: https://github.com/facebookresearch/esm/discussions/514.  Investigate later.
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t36_3B_UR50D.pt')
            self.repr_layer = 36
            self.embedding_size = 2560
        elif self.num_params == '15b':
            self.model, _ = esm.pretrained.esm2_t48_15B_UR50D()
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t48_15B_UR50D.pt')
            self.repr_layer = 48
            self.embedding_size = 5120
        elif 'official' in self.num_params:
            if '650m' in self.num_params:
                model_name = "facebook/esm2_t33_650M_UR50D"
                self.repr_layer = 33
                self.embedding_size = 1280
            elif '3b' in self.num_params:
                model_name = "facebook/esm2_t36_3B_UR50D"
                self.repr_layer = 36
                self.embedding_size = 2560
            elif '15b' in self.num_params:
                model_name = "facebook/esm2_t48_15B_UR50D"
                self.repr_layer = 48
                self.embedding_size = 5120
            elif '150m' in self.num_params:
                model_name = "facebook/esm2_t30_150M_UR50D"
                self.repr_layer = 30
                self.embedding_size = 640
            else:
                raise ValueError("Invalid number of parameters for ESM '{}'".format(self.num_params))

            if lora_parameters == 'attn':
                target_lora_modules = ["query", "key", "value"]
            elif lora_parameters == 'mlp':
                target_lora_modules = ["dense"]
            else:
                target_lora_modules = ["query", "key", "value", "dense"]

            if not use_task_spc_lora:

                peft_config = LoraConfig(
                    task_type=TaskType.TOKEN_CLS,
                    inference_mode=False,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_lora_modules, # also try "dense_h_to_4h" and "dense_4h_to_h"
                    lora_dropout=0.1,
                    bias="none" # or "all" or "lora_only"
                )

                if use_q_lora and not use_lora:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit = True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                else:
                    bnb_config = None

                self.model = EsmForMaskedLMQuant.from_pretrained(model_name, quantization_config=bnb_config)
                self.model = set_attention_type(self.model, protein_attention_type)


                if use_lora and not use_q_lora:
                    self.model = get_peft_model(self.model, peft_config)
                elif use_q_lora:

                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

                    self.model = get_peft_model(self.model, peft_config)
            else:
                peft_config = MoLoRAConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_lora_modules,
                    lora_dropout=0.1,
                    bias='none',
                    task_type="CAUSAL_LM",
                    moe_num_experts=lora_num
                )
                if use_q_lora and not use_lora:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit = True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                else:
                    bnb_config = None
                self.model = EsmForMaskedLMQuant.from_pretrained(model_name, quantization_config=bnb_config)
                self.model = set_attention_type(self.model, protein_attention_type)

                if use_lora and not use_q_lora:
                    self.model = get_moepeft_model(self.model, peft_config)
                elif use_q_lora:

                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

                    self.model = get_moepeft_model(self.model, peft_config)

        else:

            raise ValueError(f'ESM model with {self.num_params} parameters is not implemented')

        if self.use_prefix:
            self.prefix_model = ESMPrefix(self.repr_layer, self.model.attention_heads, self.embedding_size, prefix_dropout, prefix_attn_bn, prefix_mid_dim=prefix_mid_dim)

    def set_prot_lora_group(self, index):
        set_lora_group(self.model, index)

    def forward(self, tokens, aggregate = True):
        # Modified forward from ESM_PLM_basic
        # ESM API to process forward pass
        # IF aggregate=True, return is shape (B,E), else (B,len,E)

        # Split into chunks here ------:
        batch_tokens, batch_keys, eos_loc = self.seq_proc(tokens)
        # batch_keys will be None if we don't need to split the tokens (allows you to detect when reverse splitting is needed)
        if self.use_prefix:
            bs = batch_tokens.shape[0]
            prefix_states = self.prefix_model(bs)
        else:
            prefix_states = {'self': None}
        if self.max_batch_forward_pass is not None:
            # Restrict the batch size of a forward pass by the user-given parameter
            res_list = []
            split_tokens = torch.split(batch_tokens, self.max_batch_forward_pass, dim = 0)
            for i in range(len(split_tokens)):
                if "official" in self.num_params:
                    r = self.model(batch_tokens, output_hidden_states = True)
                    r['representations'] = r['hidden_states']
                else:
                    r = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts = False)
                res_list.append(r)
            results = concat_tensor_dict(res_list)
        else:
            # FIXME
            # Everything passed in one batch
            if "official" in self.num_params:
                results = self.model(batch_tokens, output_hidden_states = True)
                results['representations'] = results['hidden_states']
            else:
                results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts = False)

        z = results['representations'][self.repr_layer]
        #z = results['representations'][self.repr_layer][:,:245,:] # For testing
        # print(results['representations'].keys())
        # z = results['representations'][-1]

        if aggregate:
            # Reduce to per-sequence token through mean:
            padmask = (batch_tokens == self.padding_idx)
            z = self.pooler(z, batch_keys = batch_keys, padding_mask = padmask)
            logits = results['logits']
            # WARNING: Pooling with split long_protein_strategy will pool all extra CLS and EOS tokens across pooler
        else: # If not aggregating, don't touch logits
            # Map split sequences back to original size:
            if batch_keys is not None:
                z = reverse_batched_split(z, batch_keys, eos_locs = eos_loc)
                logits = reverse_batched_split(results['logits'], batch_keys, eos_locs = eos_loc)
            else:
                logits = results['logits']

        #print('logits', results['logits'].shape)
        return z, logits # Logits are for masked language modeling