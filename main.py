import jax.numpy as jnp
from jax import jit, vmap   # , grad

from matplotlib import pyplot as plt
from imax import transforms
from PIL import Image
from time import time



def main():
    # img1 = np.ones((240, 240, 3), dtype=np.uint8) * 255
    # img2 = np.zeros((240, 240, 3), dtype=np.uint8)
    # img = blend(img1, img2, 0.5)
    # plt.imshow(img)
    # plt.show()

    image = Image.open('test/test.jpeg').convert('RGBA')

    # times = []
    #
    # for i in range(1):
    #     t0 = time()
    #     transformaed = image.rotate(45)
    #     times.append(time() - t0)
    #
    # print(jnp.mean(jnp.array(times)))
    # print(jnp.median(jnp.array(times)))

    image = jnp.asarray(Image.open('test/test.jpeg').convert('RGBA')).astype('float32')
    images = jnp.tile(jnp.expand_dims(image, 0), [64, 1, 1, 1])

    print(image.shape)

    times = []

    T = transforms.scale(cx=1, cy=1)()
    T = transforms.rotate(rz=1)(T)

    t0 = time()
    transformed_image = jit(transforms.apply_transforms)(image,
                                                         T,
                                                         mask_value=-1,  # jnp.array([0, 0, 0, 255])
                                                         )
    print(time() - t0)

    for _ in range(100):
        t0 = time()
        transformed_image = jit(transforms.apply_transforms)(image,
                                                             T,
                                                             mask_value=-1)
        times.append(time() - t0)

    print(jnp.mean(jnp.array(times)))
    print(jnp.median(jnp.array(times)))

    plt.imshow(transformed_image)
    plt.show()

    #run with vmap

    images = jnp.tile(jnp.expand_dims(image, 0), [64, 1, 1, 1])
    Ts = jnp.tile(jnp.expand_dims(T, 0), [64, 1, 1])
    mask_values = jnp.tile(-1, [64])
    print(images.shape)

    times = []

    t0 = time()
    transformed_image = vmap(jit(transforms.apply_transforms))(images,
                                                         Ts,
                                                         mask_value=mask_values,  # jnp.array([0, 0, 0, 255])
                                                         )
    print(time() - t0)

    for _ in range(100):
        t0 = time()
        transformed_image = vmap(jit(transforms.apply_transforms))(images,
                                                             Ts,
                                                             mask_value=mask_values)  # jnp.array([0, 0, 0, 255]))
        times.append(time() - t0)

    print(jnp.mean(jnp.array(times)))
    print(jnp.median(jnp.array(times)))

    plt.imshow(transformed_image[0])
    plt.show()


if __name__ == '__main__':
    main()
