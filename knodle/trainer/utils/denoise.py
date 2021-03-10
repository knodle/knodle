from tqdm import tqdm

import numpy as np
import scipy.sparse as ss


def activate_neighbors(
        rule_matches_z: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """
    Find all closest neighbors and take the same label ids
    Args:
        rule_matches_z: All rule matches. Shape: instances x rules
        indices: Neighbor indices from knn (evtl. a generator object). If no indices provided for an object, no activation is done.
    Returns:
    """
    #print("rm", rule_matches_z.shape, rule_matches_z.dtype)
    if isinstance(rule_matches_z, ss.csr_matrix):
        new_z_matrix = ss.lil_matrix(rule_matches_z.shape, dtype=np.int8)
    else:
        new_z_matrix = np.zeros_like(rule_matches_z)
    
    # make sure initial matches are preserved
    new_z_matrix[rule_matches_z != 0] = 1
    print("pre", rule_matches_z.sum(), new_z_matrix.sum())
    
    for index, neighbors in enumerate(tqdm(indices)):
        if len(neighbors) == 0:
            continue
            
        neighbors = neighbors.astype(np.int64)
        #print(neighbors.shape, neighbors.dtype)
        neighborhood_z = rule_matches_z[neighbors, :].astype(np.int8)
        #print(neighborhood_z.shape, neighborhood_z.dtype)
        
        activated_lfs = neighborhood_z.sum(axis=0)  # Add all lf activations
        #print("actlfs", activated_lfs.shape, activated_lfs.dtype)

        # All with != 0 are valid. We could also do some division here for weighted
        activated_lfs = np.where(activated_lfs > 0)[0]
        #print("actlfs", activated_lfs.shape, activated_lfs.dtype)
        
        new_z_matrix[index, activated_lfs] = 1
        
    # check out the amount of new matches
    print("post", rule_matches_z.sum(), new_z_matrix.sum())
    
    # make sure all old matches persist
    assert new_z_matrix[rule_matches_z == 1].sum() == rule_matches_z[rule_matches_z == 1].sum() 
    
    if isinstance(new_z_matrix, ss.lil_matrix):
        return new_z_matrix.tocsr()
    else:
        return new_z_matrix
