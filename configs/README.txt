# unified_dataset
********DMPNN*********
batch_size : 16
lr : 0.0001435
depth : 5
dropout : 0.2
graph_pool : max
hidden_size : 600
mlp_hidden_size : 900
********Tetra_DMPNN*********
batch_size : 32
depth : 3
dropout : 0.2
graph_pool : max
hidden_size : 600
lr : 0.0002267
mlp_hidden_size : 900
********CHIRO*************
EConv_mlp_hidden_layer_number : 1
EConv_mlp_hidden_size : 32
F_H : 8
F_z : 64
GAT_N_heads : 8
GAT_hidden_layer_number : 3
GAT_hidden_node_size : 64
batch_size : 16
dropout : 0
encoder_hidden_layer_number : 2
encoder_hidden_size : 32
encoder_reduction : sum
encoder_sinusoidal_shift_hidden_layer_number : 2
encoder_sinusoidal_shift_hidden_size : 128
lr : 0.0006232
output_mlp_hidden_layer_number : 1
output_mlp_hidden_size : 128

# larger_dataset
**********DMPNN************
batch_size : 32
depth : 2
dropout : 0
graph_pool : max
hidden_size : 1200
lr : 0.0001051
mlp_hidden_size : 900
**********Tetra_DMPNN******
batch_size : 32
depth : 4
dropout : 0.2
graph_pool : max
hidden_size : 900
lr : 0.000588
mlp_hidden_size : 600
**********CHIRO***********
EConv_mlp_hidden_layer_number : 2
EConv_mlp_hidden_size : 256
F_H : 64
F_z : 8
GAT_N_heads : 8
GAT_hidden_layer_number : 2
GAT_hidden_node_size : 32
batch_size : 32
dropout : 0
encoder_hidden_layer_number : 3
encoder_hidden_size : 256
encoder_reduction : sum
encoder_sinusoidal_shift_hidden_layer_number : 2
encoder_sinusoidal_shift_hidden_size : 256
lr : 0.0002695
output_mlp_hidden_layer_number : 3
output_mlp_hidden_size : 64
