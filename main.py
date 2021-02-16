import random as orandom
from time import time

import jax
import jax.numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

from imax import transforms
from imax import color_transforms
from imax import randaugment


def main():
    # img1 = np.ones((240, 240, 3), dtype=np.uint8) * 255
    # img2 = np.zeros((240, 240, 3), dtype=np.uint8)
    # img = blend(img1, img2, 0.5)
    # plt.imshow(img)
    # plt.show()

    random_key = random.PRNGKey(orandom.randint(0, 1000))
    image1 = jnp.asarray(Image.open('tests/test.jpeg').convert('RGBA')).astype('uint8')
    image2 = jnp.asarray(Image.open('tests/test.jpeg').convert('RGBA').rotate(90)).astype('uint8')

    images1 = jnp.tile(jnp.expand_dims(image1, 0), [64, 1, 1, 1])
    num_layers = jnp.tile(jnp.expand_dims(3, (0,)), [64])
    magnitudes = jnp.tile(jnp.expand_dims(10, (0,)), [64])

    results = []

    for _ in range(9):
        random_key, split_key = random.split(random_key, 2)
        split_keys = random.split(split_key, 64)
        # split_keys = random.randint(split_keys[0], [64], minval=0, maxval=1000000)
        t_0 = time()

        # for i in [images1, num_layers, magnitudes, split_keys]:
        #     print(i.shape)

        transformed_image = randaugment.distort_image_with_randaugment(
            image1,
            num_layers=3,
            magnitude=10,
            random_key=split_key
        )

        # transformed_images = jax.vmap(randaugment.distort_image_with_randaugment)(
        #     images1,
        #     num_layers=num_layers,
        #     magnitude=magnitudes,
        #     random_key=split_keys
        # )
        print(time() - t_0)
        print()

        results.append(transformed_image)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, results):
        # Iterating over the grid returns the Axes.
        ax.axis('off')
        ax.imshow(im)
    plt.tight_layout()
    plt.show()
    #
    # images1 = jnp.tile(jnp.expand_dims(image1, 0), [64, 1, 1, 1])
    # images2 = jnp.tile(jnp.expand_dims(image2, 0), [64, 1, 1, 1])
    #
    # cutout_mask = color_transforms.get_random_cutout_mask(random_key, image1.shape, (80, 80))
    #
    #
    #
    # t0 = time()
    #
    # transformed_ims = {
    #     'none': image1,
    #     'blend': color_transforms.blend(image1, image2, 0.5),
    #     'cutout': color_transforms.cutout(image1, cutout_mask),
    #     'solarize': color_transforms.solarize(image1),
    #     'solarize_add': color_transforms.solarize_add(image1, 100,),
    #     'color': color_transforms.color(image1, 0.2),
    #     'contrast': color_transforms.contrast(image1, 0.6),
    #     'brightness': color_transforms.brightness(image1, 0.5),
    #     'posterize1': color_transforms.posterize(image1, 1),
    #     'posterize2': color_transforms.posterize(image1, 2),
    #     'posterize4': color_transforms.posterize(image1, 4),
    #     'posterize6': color_transforms.posterize(image1, 6),
    #     'autocontrast': color_transforms.autocontrast(image1),
    #     'sharpness_0.1': color_transforms.sharpness(image1, 0.1),
    #     'sharpness_2.0': color_transforms.sharpness(image1, 2.0),
    #     'equalize': color_transforms.equalize(image1),
    #     'invert': color_transforms.invert(image1),
    # }
    # print(time() - t0)
    #
    # t0 = time()
    #
    # transformed_ims = {
    #     'none': image1,
    #     'blend': color_transforms.blend(image1, image2, 0.5),
    #     'cutout': color_transforms.cutout(image1, cutout_mask),
    #     'solarize': color_transforms.solarize(image1),
    #     'solarize_add': color_transforms.solarize_add(image1, 100, ),
    #     'color': color_transforms.color(image1, 0.2),
    #     'contrast': color_transforms.contrast(image1, 0.6),
    #     'brightness': color_transforms.brightness(image1, 0.5),
    #     'posterize1': color_transforms.posterize(image1, 1),
    #     'posterize2': color_transforms.posterize(image1, 2),
    #     'posterize4': color_transforms.posterize(image1, 4),
    #     'posterize6': color_transforms.posterize(image1, 6),
    #     'autocontrast': color_transforms.autocontrast(image1),
    #     'sharpness_0.1': color_transforms.sharpness(image1, 0.1),
    #     'sharpness_2.0': color_transforms.sharpness(image1, 2.0),
    #     'equalize': color_transforms.equalize(image1),
    #     'invert': color_transforms.invert(image1),
    # }
    # print(time() - t0)
    #
    # t0 = time()
    #
    # transformed_ims = {
    #     'none': image1,
    #     'blend': color_transforms.blend(image1, image2, 0.5),
    #     'cutout': color_transforms.cutout(image1, cutout_mask),
    #     'solarize': color_transforms.solarize(image1),
    #     'solarize_add': color_transforms.solarize_add(image1, 100, ),
    #     'color': color_transforms.color(image1, 0.2),
    #     'contrast': color_transforms.contrast(image1, 0.6),
    #     'brightness': color_transforms.brightness(image1, 0.5),
    #     'posterize1': color_transforms.posterize(image1, 1),
    #     'posterize2': color_transforms.posterize(image1, 2),
    #     'posterize4': color_transforms.posterize(image1, 4),
    #     'posterize6': color_transforms.posterize(image1, 6),
    #     'autocontrast': color_transforms.autocontrast(image1),
    #     'sharpness_0.1': color_transforms.sharpness(image1, 0.1),
    #     'sharpness_2.0': color_transforms.sharpness(image1, 2.0),
    #     'equalize': color_transforms.equalize(image1),
    #     'invert': color_transforms.invert(image1),
    # }
    # print(time() - t0)
    #
    #
    # for name, im in transformed_ims.items():
    #     plt.imshow(im)
    #     plt.title(name)
    #     plt.show()

    # image = jnp.asarray(Image.open('tests/tests.jpeg').convert('RGBA')).astype('float32')
    # images = jnp.tile(jnp.expand_dims(image, 0), [64, 1, 1, 1])
    #
    # print(image.shape)
    #
    # times = []
    #
    # T = transforms.scale(cx=1, cy=1)()
    # T = transforms.rotate(rz=1)(T)
    #
    # t0 = time()
    # transformed_image = jit(transforms.apply_transforms)(
    #     image,
    #     T,
    #     mask_value=-1,  # jnp.array([0, 0, 0, 255])
    # )
    # print(time() - t0)
    #
    # for _ in range(100):
    #     t0 = time()
    #     transformed_image = jit(transforms.apply_transforms)(image,
    #                                                          T,
    #                                                          mask_value=-1)
    #     times.append(time() - t0)
    #
    # print(jnp.mean(jnp.array(times)))
    # print(jnp.median(jnp.array(times)))
    #
    # plt.imshow(transformed_image)
    # plt.show()
    #
    # #run with vmap
    #
    # images = jnp.tile(jnp.expand_dims(image, 0), [64, 1, 1, 1])
    # Ts = jnp.tile(jnp.expand_dims(T, 0), [64, 1, 1])
    # mask_values = jnp.tile(-1, [64])
    # print(images.shape)
    #
    # times = []
    #
    # t0 = time()
    # transformed_image = vmap(jit(transforms.apply_transforms))(
    #     images,
    #     Ts,
    #     mask_value=mask_values,  # jnp.array([0, 0, 0, 255])
    # )
    # print(time() - t0)
    #
    # for _ in range(100):
    #     t0 = time()
    #     transformed_image = vmap(jit(transforms.apply_transforms))(
    #         images,
    #         Ts,
    #         mask_value=mask_values)  # jnp.array([0, 0, 0, 255]))
    #     times.append(time() - t0)
    #
    # print(jnp.mean(jnp.array(times)))
    # print(jnp.median(jnp.array(times)))
    #
    # plt.imshow(transformed_image[0])
    # plt.show()


if __name__ == '__main__':
    main()
