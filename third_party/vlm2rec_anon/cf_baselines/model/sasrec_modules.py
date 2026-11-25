import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class FeedForward(nn.Module):

    def __init__(self,args):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.inner_size)
        self.intermediate_act_fn = self.get_hidden_act(args.act_fn)

        self.dense_2 = nn.Linear(args.inner_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
    
    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": F.gelu,
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states

class MultiHeadAttention(nn.Module):

    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.args = args
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(args.attn_dropout_prob)
        
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, input_tensor, attention_mask):            
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

### Vanilla Transformer Layer ###
class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.args = args
        self.attention_layer = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output= self.attention_layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        
        return feedforward_output
    
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        layer = Layer(args) # self attention
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states
