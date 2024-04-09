from jax import numpy as jnp


def compare(inputs, targets, outputs):
    try:
        assert jnp.allclose(targets, outputs, atol=1e-4)
    except AssertionError as ex:
        for k, v in zip(['inputs', 'outputs', 'target'],
                        [inputs, outputs, targets]):
            print(k)
            v = jnp.transpose(v, (2, 0, 1))
            for channel in v:
                print(channel)
            print()
        raise ex
