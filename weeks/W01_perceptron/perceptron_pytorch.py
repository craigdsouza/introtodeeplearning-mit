import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = {
    'AND': torch.tensor([[0],[0],[0],[1]], dtype=torch.float32),
    'OR': torch.tensor([[0],[1],[1],[1]], dtype=torch.float32),
    'XOR': torch.tensor([[0],[1],[1],[0]], dtype=torch.float32),
}


class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,1)

    def forward(self,x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

################################
#  Ex 3  Perceptron            #
################################

# model = Perceptron()
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.5)

# for key, labels in y.items():
#     print(f"\n-----Training for {key} gate with single perceptron-----")
#     for epoch in range(5000):
#         y_hat = model(X)
#         loss = loss_fn(y_hat, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if epoch % 500 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item():.3f}")

#     # Inference
#     print(f"\n-----Inference for {key} gate with single perceptron-----")
#     with torch.no_grad():
#         probs = torch.sigmoid(model(X))
#         print(probs)

################################
#  Ex 3  Two Layer NN          #
################################
# model = TwoLayerNet()
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.5)


# print(f"\n-----Training for XOR gate with two layer NN-----")
# labels = y['XOR']

# for epoch in range(5000):
#     y_hat = model(X)
#     loss = loss_fn(y_hat, labels)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item():.3f}")

# # Inference
# print(f"\n-----Inference for {key} gate with two layer NN-----")
# with torch.no_grad():                      # disable gradient tracking for inference
#     probs = torch.sigmoid(model(X))        # apply sigmoid to get probabilities
#     print(probs)

##################################
# Ex 5 Perceptron - Varying LR   #
##################################

loss_fn = nn.BCEWithLogitsLoss()
labels = y['AND']
lrs = [0.001, 0.1, 10.0]
all_losses = {}

for lr in lrs:
    model = Perceptron()   # fresh model each run — weights must not carry over
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []

    print(f"\n-----Training AND gate, lr={lr}-----")
    for epoch in range(500):
        y_hat = model(X)
        loss = loss_fn(y_hat, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.3f}")

    all_losses[lr] = losses

    with torch.no_grad():
        probs = torch.sigmoid(model(X))
    print(f"Final probs: {probs.squeeze().tolist()}")

plt.figure(figsize=(8, 5))
for lr, losses in all_losses.items():
    plt.plot(losses, label=f'lr={lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Rate Sensitivity — AND gate')
plt.legend()
plt.tight_layout()
plt.savefig('weeks/W01_perceptron/lr_sensitivity.png')
plt.show()
print("Plot saved to weeks/W01_perceptron/lr_sensitivity.png")
