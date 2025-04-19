import matplotlib.pyplot as plt

# Extracted data from the training logs
epochs = list(range(1, 21))  # Epochs 1 to 20
train_loss = [3.8422, 1.7863, 1.7000, 1.4351, 1.3648, 1.0616, 1.1423, 1.0210, 0.7908, 0.7750, 
              0.6412, 0.7104, 0.5773, 0.5151, 0.4937, 0.4796, 0.4552, 0.4437, 0.4404, 0.4294]
val_loss = [1.9531, 1.7378, 1.5243, 1.5082, 1.0579, 1.2512, 1.1506, 0.8764, 0.9250, 0.6840, 
            0.8920, 0.7251, 0.6414, 0.6255, 0.6217, 0.5968, 0.5889, 0.5852, 0.5953, 0.5844]
train_accuracy = [0.2030, 0.3425, 0.2907, 0.4065, 0.4266, 0.5957, 0.5693, 0.6513, 0.8035, 0.8053, 
                  0.8576, 0.8044, 0.8756, 0.9000, 0.9064, 0.9087, 0.9131, 0.9164, 0.9159, 0.9180]
val_accuracy = [0.3320, 0.2892, 0.3502, 0.3546, 0.6096, 0.5228, 0.5543, 0.7582, 0.7104, 0.8447, 
                0.7295, 0.8006, 0.8466, 0.8566, 0.8541, 0.8610, 0.8642, 0.8653, 0.8619, 0.8668]

# Plot Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')
plt.close()

# Plot Accuracy Curves
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy', color='green')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curves.png')
plt.close()

print("Loss and accuracy curves have been saved as 'loss_curves.png' and 'accuracy_curves.png' respectively.")