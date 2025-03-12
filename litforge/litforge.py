
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleLLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.load_weights(model_name)

    def load_weights(self, model_name):
        # Load weights, tokenizer, and config using transformers
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = model.config

    def forward(self, input_ids, past_key_values=None):
        # Convert input IDs to embeddings
        x = self.weights["model.embed_tokens.weight"][input_ids]
        new_past = []
        for i in range(self.config.num_hidden_layers):

            q = F.linear(x, self.weights[f"model.layers.{i}.self_attn.q_proj.weight"])
            k = F.linear(x, self.weights[f"model.layers.{i}.self_attn.k_proj.weight"])
            v = F.linear(x, self.weights[f"model.layers.{i}.self_attn.v_proj.weight"])
            # Use cached keys/values if available
            if past_key_values is not None:
                past_k, past_v = past_key_values[i]
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            new_past.append((k, v))
            # Scaled dot-product attention
            scale = self.config.hidden_size ** 0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_probs, v)
            # Output projection with residual connection
            proj = F.linear(attn_out, self.weights[f"model.layers.{i}.self_attn.o_proj.weight"])
            x = x + proj
            # Feed-forward network with GELU activation and residual connection
            ff = F.linear(x, self.weights[f"model.layers.{i}.mlp.fc_in.weight"],
                           self.weights.get(f"model.layers.{i}.mlp.fc_in.bias", None))
            ff = F.gelu(ff)
            ff = F.linear(ff, self.weights[f"model.layers.{i}.mlp.fc_out.weight"],
                           self.weights.get(f"model.layers.{i}.mlp.fc_out.bias", None))
            x = x + ff
        # Final layer normalization and output logits via weight tying
        x = F.layer_norm(x, (self.config.hidden_size,),
                         weight=self.weights["model.norm.weight"],
                         bias=self.weights["model.norm.bias"])
        logits = torch.matmul(x, self.weights["model.embed_tokens.weight"].T)
        return logits, new_past

    def generate(self, prompt, max_length=512):
        # Tokenize prompt and generate text token-by-token
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        self.eval()
        with torch.no_grad():
            logits, past = self.forward(input_ids, None)
            for _ in range(max_length - input_ids.shape[1]):
                last_token = input_ids[:, -1:]
                logits, past = self.forward(last_token, past_key_values=past)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(input_ids[0].tolist())
