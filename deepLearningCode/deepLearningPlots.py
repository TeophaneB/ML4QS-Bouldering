import matplotlib.pyplot as plt
import seaborn as sns




epochs = list(range(1, 33))

train_loss = [
    1.4878, 1.1508, 1.1824, 1.3118, 1.0908, 1.0700, 1.1175, 1.0330,
    0.9802, 1.0273, 1.0447, 0.9776, 0.9053, 0.8645, 0.9417, 0.8278,
    0.7917, 0.8462, 0.8417, 0.7888, 0.8419, 0.7399, 0.7209, 0.6847,
    0.6030, 0.6902, 0.6572, 0.6711, 0.6003, 0.7693, 0.6359, 0.5368
]

val_loss = [
    1.1538, 1.1384, 1.0930, 1.0360, 1.0306, 1.0165, 0.9980, 1.0104,
    1.0167, 0.9575, 0.9440, 0.9449, 0.9325, 0.8619, 0.8746, 0.8688,
    0.8174, 0.8048, 0.8108, 0.7832, 0.7935, 0.7756, 0.7513, 0.7953,
    0.7253, 0.7402, 0.7228, 0.7050, 0.7041, 0.8032, 0.7179, 0.6770
]

plt.figure(figsize=(9, 5))

plt.plot(epochs, train_loss, marker="o", label="Training loss")
plt.plot(epochs, val_loss, marker="o", label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.legend()
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()