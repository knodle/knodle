import logging
from typing import Iterable
from tqdm import tqdm

import numpy as np
import scipy.sparse as ss

logger = logging.getLogger(__name__)


def activate_neighbors(
        rule_matches_z: np.ndarray, indices: Iterable[np.ndarray]
) -> np.ndarray:
    """
    Take provided closest neighbors and add their rule matches.
    If no indices provided for an instance, no activation is done (no matches added).
    Args:
        rule_matches_z: All rule matches. Shape: instances x rules
        indices: Neighbor indices from knn per instance.
    Returns:
    """
    if isinstance(rule_matches_z, ss.csr_matrix):
        new_z_matrix = ss.lil_matrix(rule_matches_z.shape, dtype=np.int8)
    else:
        new_z_matrix = np.zeros_like(rule_matches_z)
    
    # make sure initial matches are preserved
    new_z_matrix[rule_matches_z != 0] = 1

    for index, neighbors in enumerate(tqdm(indices)):
        if len(neighbors) == 0:
            continue
            
        neighbors = neighbors.astype(np.int64)
        neighborhood_z = rule_matches_z[neighbors, :].astype(np.int8)

        activated_lfs = neighborhood_z.sum(axis=0)  # Add all lf activations

        # All with != 0 are valid. We could also do some division here for weighted
        activated_lfs = np.where(activated_lfs > 0)[0]

        # add neighbor matches to the target object
        new_z_matrix[index, activated_lfs] = 1
        
    # check out the amount of new matches
    logger.info(f"Match increase: {new_z_matrix.sum() / rule_matches_z.sum() * 100 - 100:.2f}%")
    
    if isinstance(new_z_matrix, ss.lil_matrix):
        return new_z_matrix.tocsr()
    else:
        return new_z_matrix
