import matplotlib.pyplot as plt

from src.api_image_vq import api_train_vq

if __name__ == '__main__':
    before, after = api_train_vq(64)
    plt.imshow(before)
    plt.savefig('./data/generated/before.png')
    plt.imshow(after)
    plt.savefig('./data/generated/after.png')
