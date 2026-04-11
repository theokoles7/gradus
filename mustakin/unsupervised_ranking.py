def correlation_based_ranking(batch_attributes):
    """
    Implement the correlation-based ranking algorithm
    """
    if len(batch_attributes) < 2:
        return list(batch_attributes.keys())
    
    batch_indices = list(batch_attributes.keys())
    attributes_matrix = np.array([batch_attributes[idx] for idx in batch_indices])
    m = len(batch_indices)  # number of batches
    n = attributes_matrix.shape[1]  # number of attributes (3: mean, std, skewness)
    
    # Initialize weights
    W = np.zeros(m)
    
    # Implement the ranking algorithm
    for i in range(m):
        for j in range(i + 1, m):
            # Calculate correlation coefficient between batch i and batch j
            corr_coeff, _ = pearsonr(attributes_matrix[i], attributes_matrix[j])
            
            # Handle NaN correlation (when std is 0)
            if np.isnan(corr_coeff):
                corr_coeff = 0.0
            
            P_ij = corr_coeff
            
            if P_ij >= 0:
                if W[i] >= 0 and W[j] >= 0:
                    W[i] += P_ij
                    W[j] += P_ij
                elif W[i] < 0 and W[j] < 0:
                    W[i] -= P_ij
                    W[j] -= P_ij
                elif W[i] >= 0 and W[j] < 0:
                    W[i] -= P_ij
                    W[j] += P_ij
                else:  # wi < 0 and wj >= 0
                    W[i] += P_ij
                    W[j] -= P_ij
            else:  # P_ij < 0
                if W[i] >= 0 and W[j] >= 0:
                    if W[i] < W[j]:
                        W[i] += P_ij
                        W[j] -= P_ij
                    else:
                        W[i] -= P_ij
                        W[j] += P_ij
                elif W[i] < 0 and W[j] < 0:
                    if W[i] < W[j]:
                        W[i] += P_ij
                        W[j] -= P_ij
                    else:
                        W[i] -= P_ij
                        W[j] += P_ij
                elif W[i] >= 0 and W[j] < 0:
                    W[i] -= P_ij
                    W[j] += P_ij
                else:  # wi < 0 and wj >= 0
                    W[i] += P_ij
                    W[j] -= P_ij
    
    # Sort batches by weights (descending order - higher weight = higher rank)
    ranked_indices = [batch_indices[i] for i in np.argsort(W)[::-1]]
    
    return ranked_indices, W