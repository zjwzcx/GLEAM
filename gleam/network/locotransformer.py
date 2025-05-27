import torch
import torch.nn as nn
import torch.nn.functional as F
import gleam.network.init as init


class LocoTransformer_GLEAM(nn.Module):
    def __init__(
        self,
        encoder,
        output_shape,
        state_input_shape,
        visual_input_shape,
        transformer_params=[],
        append_hidden_shapes=[],
        append_hidden_init_func=init.basic_init,
        net_last_init_func=init.uniform_init,
        activation_func=nn.ReLU,
        add_ln=False,
        detach=False,
        state_detach=False,
        max_pool=False,
        token_norm=False,
        use_pytorch_encoder=False,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        
        # Add value embedding layer to map {0,1,2,3} to feature space
        self.value_embedding = nn.Embedding(
            num_embeddings=4,  # Original values: -1,0,1,2 → shifted to 0,1,2,3
            embedding_dim=encoder.in_channels
        )
        
        # Initialize embedding weights to emphasize value 2 (shifted to index 3)
        with torch.no_grad():
            # Increase weight magnitude for index 3 (original value 2)
            self.value_embedding.weight[3] *= 3.0  
            # Optional: Keep other embeddings smaller
            self.value_embedding.weight[:3] *= 0.5

        self.add_ln = add_ln
        self.detach = detach
        self.state_detach = state_detach

        self.state_input_shape = state_input_shape
        self.visual_input_shape = visual_input_shape
        self.activation_func = activation_func

        self.max_pool = max_pool
        visual_append_input_shape = self.encoder.visual_dim

        self.token_norm = token_norm
        if self.token_norm:
            self.token_ln = nn.LayerNorm(self.encoder.visual_dim)
            self.state_token_ln = nn.LayerNorm(self.encoder.visual_dim)

        self.use_pytorch_encoder = use_pytorch_encoder
        if not self.use_pytorch_encoder:
            self.visual_append_layers = nn.ModuleList()
            for n_head, dim_feedforward in transformer_params:
                visual_att_layer = nn.TransformerEncoderLayer(
                    visual_append_input_shape, n_head, dim_feedforward, dropout=0
                )
                self.visual_append_layers.append(visual_att_layer)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                self.encoder.visual_dim, 
                transformer_params[0][0], 
                transformer_params[0][1], 
                dropout=0
            )
            encoder_norm = nn.LayerNorm(self.encoder.visual_dim)
            self.visual_trans_encoder = nn.TransformerEncoder(
                encoder_layer, 
                len(transformer_params), 
                encoder_norm
            )

        self.per_modal_tokens = self.encoder.per_modal_tokens
        self.second = False if self.encoder.in_channels in [4,12] else True

        # Feature processing layers
        self.visual_append_fcs = []
        visual_append_input_shape = visual_append_input_shape * 2
        if self.second:
            visual_append_input_shape += self.encoder.visual_dim
            
        for next_shape in append_hidden_shapes:
            visual_fc = nn.Linear(visual_append_input_shape, next_shape)
            append_hidden_init_func(visual_fc)
            self.visual_append_fcs.append(visual_fc)
            self.visual_append_fcs.append(self.activation_func())
            if self.add_ln:
                self.visual_append_fcs.append(nn.LayerNorm(next_shape))
            visual_append_input_shape = next_shape

        visual_last = nn.Linear(visual_append_input_shape, output_shape)
        net_last_init_func(visual_last)
        self.visual_append_fcs.append(visual_last)
        self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)

        self.normalizer = None

    def forward(self, x):
        """Process input with value-specific feature enhancement.
        Args:
            state_input: [N, state_input_shape]
            visual_input: [N, visual_input_shape]
        Returns:
            out: [N, output_shape]
        """
        # Split state and visual inputs
        state_input = x[..., :self.state_input_shape]
        visual_input = x[..., self.state_input_shape:].view(
            torch.Size(state_input.shape[:-1] + self.visual_input_shape)
        )
        
        # Shift values to [0,3] range and embed
        shifted_visual = (visual_input + 1).long()  # Map [-1,0,1,2]→[0,1,2,3]
        embedded_visual = self.value_embedding(shifted_visual).squeeze(-1)  # [N, H, W, C]
        
        # # Permute to channel-first format (N, C, H, W)
        # embedded_visual = embedded_visual.permute(0, 3, 1, 2)
        
        # Pass through encoder
        visual_out, state_out = self.encoder(
            embedded_visual, 
            state_input, 
            detach=self.detach
        )
        
        # # Generate attention mask for original value 2 positions
        # value_mask = (visual_input == 2).float()  # (N,1,H,W)

        # # Resize mask to match feature dimensions
        # value_mask = F.interpolate(
        #     value_mask, 
        #     size=visual_out.shape[-2:], 
        #     mode='nearest'
        # )

        # # Enhance features at value 2 positions
        # visual_out = visual_out * (1 + value_mask)

        # processing pipeline
        if visual_out.shape[0] < (1 + 2 * self.per_modal_tokens):
            self.per_modal_tokens = visual_out.shape[0] // 2

        if self.token_norm:
            visual_out = self.token_ln(visual_out)
            
        if not self.use_pytorch_encoder:
            for att_layer in self.visual_append_layers:
                visual_out = att_layer(visual_out)
        else:
            visual_out = self.visual_trans_encoder(visual_out)

        # Feature aggregation
        out_state = visual_out[0, ...]
        out_first = visual_out[1:1 + self.per_modal_tokens, ...].mean(dim=0)
        out_list = [out_state, out_first]

        if self.second:
            if visual_out.shape[0] < (1 + 2 * self.per_modal_tokens):
                out_second = visual_out[1 + self.per_modal_tokens:, ...]
            else:
                out_second = visual_out[
                    1 + self.per_modal_tokens:1 + 2 * self.per_modal_tokens, ...
                ]
            out_second = out_second.mean(dim=0)
            out_list.append(out_second)

        # Final processing
        out = torch.cat(out_list, dim=-1)
        return self.visual_seq_append_fcs(out)