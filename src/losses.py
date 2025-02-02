import torch
import torch.nn as nn


class BaseLoss:
    eps = 1e-9
    
class DDCLoss(BaseLoss):
    """
    Adapted From: https://github.com/DanielTrosten/mvc/blob/main/src/lib/loss.py
    """

    def __init__(self, n_clusters):
        #super().__init__()
        self.n_output = n_clusters
        #self.eye = torch.eye(projector_n_out, device='cpu')
        self.prob_layer = torch.nn.Softmax(dim=1)
    
    @staticmethod
    def triu(X):
        """\ 
        Sum of strictly upper triangular part.
        """
        return torch.sum(torch.triu(X, diagonal=1))

    @staticmethod
    def _atleast_epsilon(X, eps=1e-9):
        """
        Ensure that all elements are >= `eps`.
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    @staticmethod
    def d_cs(A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(
            torch.diagonal(nom), 0
        )

        nom = DDCLoss._atleast_epsilon(nom)
        dnom_squared = DDCLoss._atleast_epsilon(dnom_squared, eps=BaseLoss.eps ** 2)

        d = (
            2
            / (n_clusters * (n_clusters - 1))
            * DDCLoss.triu(nom / torch.sqrt(dnom_squared))
        )
        return d

    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=BaseLoss.eps):
        """\
        Compute a Gaussian kernel matrix from a distance matrix.
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(-dist / (2 * sigma2))
        return k

    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=BaseLoss.eps):
        """\
        Compute a Gaussian kernel matrix from a distance matrix.
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = nn.functional.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(-dist / (2 * sigma2))
        return k

    @staticmethod
    def cdist(X, Y):
        """\
        Pairwise distance between rows of X and rows of Y.
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    @staticmethod
    def vector_kernel(x, rel_sigma=0.15):
        """\
        Compute a kernel matrix from the rows of a matrix.
        """
        return DDCLoss.kernel_from_distance_matrix(DDCLoss.cdist(x, x), rel_sigma)

    def __call__(self, output_hidden, output_cluster):
        cluster_outputs = self.prob_layer(output_cluster)
        hidden_kernel = DDCLoss.vector_kernel(output_hidden)
        # L_1 loss
        loss = DDCLoss.d_cs(cluster_outputs, hidden_kernel, self.n_output)

        # L_3 loss
        eye = torch.eye(self.n_output, device=output_cluster.device) #####
        m = torch.exp(-DDCLoss.cdist(cluster_outputs, eye))
        loss += DDCLoss.d_cs(m, hidden_kernel, self.n_output)

        return loss

class SelfEntropyLoss(BaseLoss):
    """
    Entropy regularization to prevent trivial solution.
    """

    def __init__(self):
        super().__init__()
        self.prob_layer = torch.nn.Softmax(dim=1)

    def __call__(self, cluster_output):
        eps = 1e-8

        cluster_output = self.prob_layer(cluster_output)
        prob_mean = cluster_output.mean(dim=0)
        prob_mean[(prob_mean < eps).data] = eps
        loss = (prob_mean * torch.log(prob_mean)).sum()

        return loss


class ClusteringLoss:

    def __init__(self, n_classes=5):
        #self.weight = 0.1
        self.ddcLoss = DDCLoss(n_classes)
        self.selfEntropyLoss = SelfEntropyLoss()

    def __call__(self, output_hidden, output_cluster, latents):
        ddcloss = self.ddcLoss(output_hidden, output_cluster)
        selfentropy = self.selfEntropyLoss(output_cluster)

        #print(ddcloss + selfentropy)
        #return 0.1*ddcloss + 0.1*selfentropy

        return ddcloss + selfentropy


"""
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def create_triplets(data, labels):
        triplets = []
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            # Indices of samples with the current label
            label_indices = torch.where(labels == label)[0]

            # Create anchor-positive pairs
            for anchor_index in label_indices:
                anchor = data[anchor_index]
                positive_index = torch.randint(0, len(label_indices), (1,)).item()
                positive = data[label_indices[positive_index]]

                # Create negative samples with a different label
                negative_label = label
                while negative_label == label:
                    negative_index = torch.randint(0, len(labels), (1,)).item()
                    negative_label = labels[negative_index]

                negative = data[negative_index]

                triplets.append((anchor, positive, negative))

        return triplets

    def forward(self, latents, cluster_labels):
        #cluster_labels = torch.argmax(torch.nn.functional.softmax(output_cluster, dim=1), axis=1)
        triplets = self.create_triplets(latents, cluster_labels)
        anchors, positives, negatives = zip(*triplets)

        # Convert the lists to PyTorch tensors
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        distance_positive = torch.norm(anchors - positives, dim=1)
        distance_negative = torch.norm(anchors - negatives, dim=1)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(loss)

"""

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def create_triplets(data, labels):
        triplets = []
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            # Indices of samples with the current label
            label_indices = torch.where(labels == label)[0]

            # Create anchor-positive pairs
            for anchor_index in label_indices:
                anchor = data[anchor_index]
                positive_index = torch.randint(0, len(label_indices), (1,)).item()
                positive = data[label_indices[positive_index]]

                # Create negative samples with a different label
                negative_label = label
                while negative_label == label:
                    negative_index = torch.randint(0, len(labels), (1,)).item()
                    negative_label = labels[negative_index]

                negative = data[negative_index]

                triplets.append((anchor, positive, negative))

        return triplets

    def forward(self, output_cluster1, output_cluster2, output_cluster3):
        cluster_labels1 = torch.argmax(torch.nn.functional.softmax(output_cluster1, dim=1), axis=1)
        cluster_labels2 = torch.argmax(torch.nn.functional.softmax(output_cluster2, dim=1), axis=1)
        cluster_labels3 = torch.argmax(torch.nn.functional.softmax(output_cluster3, dim=1), axis=1)
        
        triplets1 = self.create_triplets(output_cluster1, cluster_labels1)
        triplets2 = self.create_triplets(output_cluster2, cluster_labels2)
        triplets3 = self.create_triplets(output_cluster3, cluster_labels3)
        
        anchors1, positives1, negatives1 = zip(*triplets1)
        anchors2, positives2, negatives2 = zip(*triplets2)
        anchors3, positives3, negatives3 = zip(*triplets3)

        # Convert the lists to PyTorch tensors
        anchors1 = torch.stack(anchors1)
        positives1 = torch.stack(positives1)
        negatives1 = torch.stack(negatives1)

        anchors2 = torch.stack(anchors2)
        positives2 = torch.stack(positives2)
        negatives2 = torch.stack(negatives2)

        anchors3 = torch.stack(anchors3)
        positives3 = torch.stack(positives3)
        negatives3 = torch.stack(negatives3)

        distance_positive1 = torch.norm(anchors1 - positives2, dim=1)
        distance_positive2 = torch.norm(anchors1 - positives3, dim=1)
        distance_negative1 = torch.norm(anchors1 - negatives2, dim=1)
        distance_negative2 = torch.norm(anchors1 - negatives3, dim=1)
        loss1 = torch.relu(distance_positive1 - distance_negative1 + self.margin)
        loss2 = torch.relu(distance_positive2 - distance_negative2 + self.margin)
        return torch.mean(loss1) + torch.mean(loss2)
