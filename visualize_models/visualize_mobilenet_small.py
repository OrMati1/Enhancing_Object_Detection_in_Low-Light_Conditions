from graphviz import Digraph

dot = Digraph(comment='MobileNetV3 Small UNet', format='png')
dot.attr(rankdir='LR')
dot.attr('node', shape='rectangle')
dot.attr('graph', nodesep='0.1', ranksep='0.2')


dot.node('Input', 'Input Image\n(3 channels, 256x256)')
dot.node('E1', 'MobileNetV3 Block 1\n(16 filters)')
dot.node('E2', 'MobileNetV3 Block 2\n(24 filters)')
dot.node('E3', 'MobileNetV3 Block 3\n(40 filters)')
dot.node('E4', 'MobileNetV3 Block 4\n(576 filters)')
dot.node('B', 'Bottleneck\n(576 filters)')

dot.node('D4', 'Decoder 4\n(256 filters)')
dot.node('D3', 'Decoder 3\n(128 filters)')
dot.node('D2', 'Decoder 2\n(64 filters)')
dot.node('D1', 'Decoder 1\n(64 filters)')
dot.node('Output', 'Output Image\n(3 channels, 256x256)')

dot.edge('Input', 'E1', label='MobileNetV3 Encoder Block 1', minlen='1')
dot.edge('E1', 'E2', label='MobileNetV3 Encoder Block 2', minlen='1')
dot.edge('E2', 'E3', label='MobileNetV3 Encoder Block 3', minlen='1')
dot.edge('E3', 'E4', label='MobileNetV3 Encoder Block 4', minlen='1')
dot.edge('E4', 'B', label='MobileNetV3 Encoder Bottleneck', minlen='1')

dot.edge('B', 'D4', label='Upsample\nPad & Concat', minlen='1')
dot.edge('D4', 'D3', label='Upsample\nPad & Concat', minlen='1')
dot.edge('D3', 'D2', label='Upsample\nPad & Concat', minlen='1')
dot.edge('D2', 'D1', label='Upsample\nPad & Concat', minlen='1')
dot.edge('D1', 'Output', label='Conv 1x1\nSigmoid', minlen='1')

dot.edge('E4', 'D4', label='Skip Connection', style='dashed', minlen='0.3')
dot.edge('E3', 'D3', label='Skip Connection', style='dashed', minlen='0.3')
dot.edge('E2', 'D2', label='Skip Connection', style='dashed', minlen='0.3')
dot.edge('E1', 'D1', label='Skip Connection', style='dashed', minlen='0.3')

dot.render('mobilenetv3_small_unet')

print("Diagram saved as 'mobilenetv3_small_unet.png'")
