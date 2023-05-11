import numpy as np

from typing import List, Tuple

from flwr.common import (
    FitRes,
    parameters_to_ndarrays,
)

from flwr.server.client_proxy import ClientProxy
import flwr as fl

class Dnc():
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.

    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(
        self, num_byzantine, *, sub_dim=10000, num_iters=1, filter_frac=1.0
    ) -> None:
        super(Dnc, self).__init__()

        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.filter_frac = filter_frac

    def dnc_aggregate(
        self, 
        results: List[Tuple[ClientProxy, FitRes]]
    ):
        updates = np.stack([parameters_to_ndarrays(result[1].parameters) for result in results])
        d = len(results)

        benign_ids = []
        for _ in range(self.num_iters):
            indices = np.random.permutation(d)[: self.sub_dim]
            sub_updates = updates[:, indices]
            mu = np.mean(sub_updates, axis=0)
            centered_update = sub_updates - mu
            _, _, v = np.linalg.svd(centered_update, full_matrices=False)
            v = v[0, :]
            s = np.array(
                [np.dot(update - mu, v) ** 2 for update in sub_updates]
            )

            good = np.argsort(s)[
                : len(updates) - int(self.filter_frac * self.num_byzantine)
            ]
            benign_ids.extend(good)

        benign_ids = list(set(benign_ids))
        benign_updates = np.mean(updates[benign_ids, :], axis=0)
        return benign_updates
