from jax import numpy as jnp


def compare(inputs, targets, outputs):
    try:
        assert jnp.allclose(targets, outputs)
    except AssertionError as ex:
        for k, v in zip(['inputs', 'outputs', 'target'],
                        [inputs, outputs, targets]):
            print(k)
            print(v)
            print()
        raise ex
