from jax import numpy as jnp


def compare(inputs, targets, outputs):
    try:
        assert jnp.allclose(targets, outputs, atol=1e-4)
    except AssertionError as ex:
        for k, v in zip(['inputs', 'outputs', 'target'],
                        [inputs, outputs, targets]):
            print(k)
            print(v)
            print()
        raise ex
