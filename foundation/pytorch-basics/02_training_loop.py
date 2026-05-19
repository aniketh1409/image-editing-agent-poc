import torch 
from torch import nn

torch.manual_seed(42)

x = torch.linspace(-10, 10, 100).unsqueeze(1) #shape (100, 1)
y = 3*x + 2

model = nn.Linear(1,1) #what does this do? creates a linear layer with 1 input feature and 1 output feature, 
#which means it will learn a weight and a bias to map from the input to the output

# print(model)

loss_fn = nn.MSELoss()
'''
predictions = model(x)
loss = loss_fn(predictions, y)

print("initial weight:", model.weight)
print("initial bias:", model.bias)

print("initial loss:", loss.item()) 

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) #lr --> learning rate, smaller = slower but careful updates
#larger = bigger but riskier updates

#weight gradient = d(loss)/d(weight)
optimizer.zero_grad()
loss.backward()
optimizer.step()

predictions = model(x)
loss = loss_fn(predictions, y)

print("Updated weight: ", model.weight)
print("Updates bias: ", model.bias)

print("final loss:", loss.item()) 
'''

#doing in a loop - the actual Training
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 300 #a 100 runs

for epoch in range(epochs):
    predictions = model(x)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0: #printing it every 10 iterations here
        print(
            f"epoch {epoch:03d} | "
            f"loss {loss.item():.4f} | "
            f"weight {model.weight.item():.4f} | " #weight has a higher gradient --> moves towards real value sooner
            f"bias {model.bias.item():.4f}" #bias takes time
        )

test_x = torch.tensor([[12.0]])
test_prediction = model(test_x)

print("test prediction for x=12:", test_prediction.item())
print("expected:", 3 * 12 + 2)
 #o/p for this run:
# epoch 000 | loss 171.3103 | weight 2.2850 | bias 0.8534
# epoch 010 | loss 0.9139 | weight 3.0000 | bias 1.0632
# epoch 020 | loss 0.6101 | weight 3.0000 | bias 1.2345
# epoch 030 | loss 0.4073 | weight 3.0000 | bias 1.3746
# epoch 040 | loss 0.2719 | weight 3.0000 | bias 1.4890
# epoch 050 | loss 0.1815 | weight 3.0000 | bias 1.5824
# epoch 060 | loss 0.1212 | weight 3.0000 | bias 1.6588
# epoch 070 | loss 0.0809 | weight 3.0000 | bias 1.7212
# epoch 080 | loss 0.0540 | weight 3.0000 | bias 1.7722
# epoch 090 | loss 0.0361 | weight 3.0000 | bias 1.8139
# epoch 100 | loss 0.0241 | weight 3.0000 | bias 1.8479
# epoch 110 | loss 0.0161 | weight 3.0000 | bias 1.8758
# epoch 120 | loss 0.0107 | weight 3.0000 | bias 1.8985
# epoch 130 | loss 0.0072 | weight 3.0000 | bias 1.9171
# epoch 140 | loss 0.0048 | weight 3.0000 | bias 1.9322
# epoch 150 | loss 0.0032 | weight 3.0000 | bias 1.9446
# epoch 160 | loss 0.0021 | weight 3.0000 | bias 1.9548
# epoch 170 | loss 0.0014 | weight 3.0000 | bias 1.9630
# epoch 180 | loss 0.0010 | weight 3.0000 | bias 1.9698
# epoch 190 | loss 0.0006 | weight 3.0000 | bias 1.9753
# epoch 200 | loss 0.0004 | weight 3.0000 | bias 1.9798
# epoch 210 | loss 0.0003 | weight 3.0000 | bias 1.9835
# epoch 220 | loss 0.0002 | weight 3.0000 | bias 1.9865
# epoch 230 | loss 0.0001 | weight 3.0000 | bias 1.9890
# epoch 240 | loss 0.0001 | weight 3.0000 | bias 1.9910
# epoch 250 | loss 0.0001 | weight 3.0000 | bias 1.9927
# epoch 260 | loss 0.0000 | weight 3.0000 | bias 1.9940
# epoch 270 | loss 0.0000 | weight 3.0000 | bias 1.9951
# epoch 280 | loss 0.0000 | weight 3.0000 | bias 1.9960
# epoch 290 | loss 0.0000 | weight 3.0000 | bias 1.9967
# test prediction for x=12: 37.99727249145508
# expected: 38



