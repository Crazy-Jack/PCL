"""Filter the clustering results by weak supervision signals"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConFilterByWeakLabels(nn.Module):
    """Filter the hard examples in clustering results by weak labels"""
    def __init__(self, temperature=0.07, base_temperature=0.07, threshold=0.1):
        super(SupConFilterByWeakLabels, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.threshold = threshold

    @torch.no_grad()
    def getLabelTileMask(self, labels, anchor_count, contrast_count, device):
        """input label indexs of a batch, return [bz, bz] binary labels
        param:
            labels: [bz, ]
        return [contrast_count * bz, contrast_count * bz] with diag all 0s
        """
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # print("mask")
        # print(mask)
        bz = labels.shape[0]
        mask = mask.repeat(anchor_count, contrast_count)
        # mask out self-contrast case
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(bz * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        return mask, logits_mask

    @torch.no_grad()
    def getHardExample(self, similarity, threshold):
        """from similarity matrix to determine which one is the hard example and build mask for it
        param:
            similarity: [2bz, 2bz] dot products
        return: [2bz, 2bz] binary mask
        """
        device = (torch.device('cuda')
                  if similarity.is_cuda
                  else torch.device('cpu'))
        hard_example_mask = (similarity < threshold) * (torch.FloatTensor([1.]).to(device))
        # print("hard example mask")
        # print(hard_example_mask)
        # print("similarity")
        # print(similarity)
        return hard_example_mask


    @torch.no_grad()
    def correct_clustering(self, hard_example_mask, cluster_mask, weak_mask, device):
        """if two data dot product is low but they are identified as the same cluster by unsupervised clustering, then it should be corrected by the weak mask"""
        mask1 = cluster_mask * hard_example_mask * weak_mask
        mask2 = (torch.FloatTensor([1.0]).to(device) - cluster_mask * hard_example_mask) * cluster_mask

        return mask1 + mask2


    def forward(self, features, cluster_labels, weak_labels, priority="supervised"):
        """input labels from clustering and labels from weak supervision, and features, \
           make sure the features that have dot product close to zero not miss labeled by the clustering results

        param:
            features: hidden vector of shape [bsz, n_views, ...]
            cluster_labels: [bz,]
            weak_labels: [bz, ]
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        contrast_count = features.shape[1]
        batch_size = features.shape[0]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(f"contrast_feature {contrast_feature}")
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Compute Dot Product
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast
        # print("logit", logits)
        # Compute Clustering Mask
        cluster_mask, logits_mask = self.getLabelTileMask(cluster_labels, anchor_count, contrast_count, device)
        # print("cluster mask")
        # print(cluster_mask)
        weak_mask, _ = self.getLabelTileMask(weak_labels, anchor_count, contrast_count, device)
        # print("weak mask")
        # print(weak_mask)

        # filters
        ## hard example mask
        hard_example_mask = self.getHardExample((logits * self.temperature).abs(), self.threshold) # [2bz, 2bz], times temp to get original similarity
        # print("Hrad")
        # print(hard_example_mask)
        ## mask operation to filter the one with higher priority
        mask = self.correct_cluster(hard_example_mask, cluster_mask, weak_mask, device)
        # print("Final mask")
        # print(mask)
        # exp logits and log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(f"loss {loss.item()}")
        return loss, (hard_example_mask * cluster_mask).sum()






if __name__ == "__main__":
    SupConFilterLoss = SupConFilterByWeakLabels()
    features = torch.ones(4, 2, 4)
    features = F.normalize(features, dim=2)
    cluster_labels = torch.LongTensor([0,1,1,2])
    weak_labels = torch.LongTensor([0,0,0,1])
    loss = SupConFilterLoss(features, cluster_labels, weak_labels)
    print("features")
    print(features)





