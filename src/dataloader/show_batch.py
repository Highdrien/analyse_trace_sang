from torch import Tensor
import matplotlib.pyplot as plt


def plot_batch(x: Tensor) -> None:

    num_images = x.shape[0]
    rows = int(num_images**0.5)
    cols = (num_images // rows) + (num_images % rows > 0)

    _, axes = plt.subplots(rows, cols, figsize=(cols, rows))

    for i in range(num_images):
        row = i // cols
        col = i % cols
        img = x[i].numpy().transpose((1, 2, 0))
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

    plt.show()
