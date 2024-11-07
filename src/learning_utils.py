import dill
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import random


def set_seed(seed: int = 0):
    """
    Sets the random number generators' seed.

    :param seed: the seed
    :type seed: int:  (Default value = 0)

    """
    random.seed(seed)
    np.random.seed(seed)


def flatten_dict(d, label=None):
    """
    Flattens a dictionary.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten_dict(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, d)


def pad_string_with_zeros(s, num_digits):
    s = str(s)
    return "0" * (num_digits - len(s)) + s


def make_update_function(model):
    def update_default(optimizer, grads, opt_state, params, batch_stats):
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        params = model.update_batch_stats(
            params,
            batch_stats,
        )
        return params, opt_state

    return update_default


def make_grokfast(model, grok_alpha=0.98, grok_lambda=2.0):
    def update_grokfast(optimizer, grads, opt_state, params, batch_stats):
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        params["grok_filter"] = jax.tree_map(
            lambda x, y: grok_alpha * x + (1 - grok_alpha) * y,
            params["grok_filter"],
            grads,
        )

        grads = jax.tree_map(
            lambda x, y: x + grok_lambda * y,
            grads,
            params["grok_filter"],
        )

        params = optax.apply_updates(params, updates)
        params = model.update_batch_stats(
            params,
            batch_stats,
        )
        return params, opt_state

    return update_grokfast


def l2_norm(params):
    return sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))


def load_model(
    learner_path: str,
):
    learner_path, checkpoint_i = learner_path.split(":")

    all_steps = {
        "pretrain": [
            filename
            for filename in sorted(os.listdir(os.path.join(learner_path, "models")))
            if filename.startswith("pretrain_")
        ],
        "default": [
            filename
            for filename in sorted(os.listdir(os.path.join(learner_path, "models")))
            if not filename.startswith("pretrain_")
        ],
    }

    key = "pretrain" if checkpoint_i.split("_")[0] == "pretrain" else "default"
    checkpoint_i = checkpoint_i.split("_")[-1]
    if checkpoint_i == "latest":
        step = all_steps[key][-1]
    else:
        step = np.argmin(
            np.abs(
                np.array(
                    [
                        int(step.split(".")[0].split("pretrain_")[-1])
                        for step in all_steps[key]
                    ]
                )
                - int(checkpoint_i)
            )
        )
        step = all_steps[key][step]
    print("Loading checkpoint {}".format(step))
    return dill.load(open(os.path.join(learner_path, "models", step), "rb"))


def maybe_restore(restore_path: str):
    loaded_model_dict = None
    if restore_path is not None:
        loaded_model_dict = load_model(restore_path)

    return loaded_model_dict
