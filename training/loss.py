import torch

import torch.nn.functional as F

# ============================================================
# NT-XENT / CONTRASTIVE LOSS
# ============================================================


def nt_xent_loss(
    z1,
    z2,
    temperature=0.07,
):

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    z1 = F.normalize(
        z1,
        dim=1,
    )

    z2 = F.normalize(
        z2,
        dim=1,
    )

    batch_size = z1.shape[0]

    # --------------------------------------------------------
    # COMBINE VIEWS
    # --------------------------------------------------------

    representations = torch.cat(
        [
            z1,
            z2,
        ],
        dim=0,
    )

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    similarity_matrix = torch.matmul(
        representations,
        representations.T,
    )

    similarity_matrix = similarity_matrix / temperature

    total_samples = 2 * batch_size

    # --------------------------------------------------------
    # REMOVE SELF-SIMILARITY
    # --------------------------------------------------------

    self_mask = torch.eye(
        total_samples,
        device=z1.device,
        dtype=torch.bool,
    )

    similarity_matrix = similarity_matrix.masked_fill(
        self_mask,
        -float("inf"),
    )

    # --------------------------------------------------------
    # POSITIVE PAIRS
    # --------------------------------------------------------

    positive_indices = (
        torch.arange(
            total_samples,
            device=z1.device,
        )
        + batch_size
    ) % total_samples

    positive_similarity = similarity_matrix[
        torch.arange(
            total_samples,
            device=z1.device,
        ),
        positive_indices,
    ]

    # --------------------------------------------------------
    # NEGATIVE MASK
    # --------------------------------------------------------

    negative_mask = torch.ones(
        (
            total_samples,
            total_samples,
        ),
        device=z1.device,
        dtype=torch.bool,
    )

    negative_mask = negative_mask & ~self_mask

    # --------------------------------------------------------
    # REMOVE POSITIVE FROM NEGATIVES
    # --------------------------------------------------------

    positive_mask = torch.zeros(
        (
            total_samples,
            total_samples,
        ),
        device=z1.device,
        dtype=torch.bool,
    )

    row_indices = torch.arange(
        total_samples,
        device=z1.device,
    )

    positive_mask[
        row_indices,
        positive_indices,
    ] = True

    negative_mask = negative_mask & ~positive_mask

    # --------------------------------------------------------
    # GET NEGATIVES
    # --------------------------------------------------------

    negatives = similarity_matrix[negative_mask].view(
        total_samples,
        total_samples - 2,
    )

    # --------------------------------------------------------
    # LOGITS
    # --------------------------------------------------------

    logits = torch.cat(
        [
            positive_similarity.unsqueeze(1),
            negatives,
        ],
        dim=1,
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    labels = torch.zeros(
        total_samples,
        dtype=torch.long,
        device=z1.device,
    )

    # --------------------------------------------------------
    # CROSS ENTROPY
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits,
        labels,
    )

    return loss
