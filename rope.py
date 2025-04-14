from typing import Tuple
import torch
import numpy as np
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_d = query.shape[-1]
    key_d = key.shape[-1]
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # ok, so together, real and imaginary create these pairs. that checks out
    theta_i = torch.arange(1, query_d //2+1,dtype=torch.float, device=device) # from the part 0 = {0i = 10000^-2(i-1)/d, i € [1,2,.., d/2]}.
    # basically going from 1 to 1/2 of d

    thetas = theta**(-2 * (theta_i - 1) / query_d) # this should be the same dimension as keys, or else attention doesnt make sese
    # following the equation on page 5
    # 0 = {0i = 10000^-2(i-1)/d, i € [1,2,.., d/2]}
    # IMPORTANT NOTE: this will only make half the thetas. For each theta, I need to do a cos and sin version

    # I think Im doing ok so far, go Darian

    # ok now I have all my thetas which will go along the diagonal
    # now I need the m and ns
    m_q = torch.arange(0, query.shape[1], dtype=torch.float, device=device) # query seq_len
    n_k = torch.arange(0, key.shape[1], dtype=torch.float, device=device) # k seq_len
    # yes I am aware that they wil be the same but the point is robustness and easiness on the eyes
    thetas_doubled = thetas
    #thetas_doubled = torch.repeat_interleave(thetas, repeats=2, dim=0) # not sure if this is efficent, but I am following the formula
    print(thetas)
    print(thetas_doubled)
    # this is the embed dim of the q and k
    angles_key = torch.einsum("i,j->ij", n_k, thetas_doubled)  # [seq_len, head_dim]


    angles_query = torch.einsum("i,j->ij", m_q, thetas_doubled)

    # this will create a matrix of their products
    # so the first row will be thetas_doubled * first m_q (0)
    # then thetas_doubled * first m_q (0) * 1
    # then thetas_doubled * first m_q (0) * 2
    # hmm. is this truely what I want
    # I feel like thetas should be increasing down, not across, and m should increase across
    # but maybe its different since Im multiplyying in a dfifferent way
    # ok yeah it checks out I think

    key_cos = angles_key.cos()
    key_sin = angles_key.sin()

    query_cos = angles_query.cos() # these 2 things (key, query) might be equal. Its my code and I get to make equal things if I want
    query_sin = angles_query.sin()

    # ok ok now we are getting to the finally!

    real_q_out = query_real * query_cos - query_imag * query_sin
    print("real_q_out: ", real_q_out)
    imag_q_out = query_real * query_sin + query_imag * query_cos
    print("imag_q_out: ", imag_q_out)
    real_k_out = key_real * key_cos - key_imag * key_sin

    imag_k_out = key_real * key_sin + key_imag * key_cos

    query_out = torch.stack([real_q_out, imag_q_out], dim=-1).reshape_as(query)
    key_out = torch.stack([real_k_out, imag_k_out], dim=-1).reshape_as(key)

    # I Am absoluely crushing it I feel like! I hope this works

    # real is correct
    # image is incorrect


    return query_out, key_out