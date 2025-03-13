class CombinedLoss(nn.Module):
    def __init__(self, triplet_margin=1.0, lambda_center=0.5, num_classes=80):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin, p=2)  # Triplet loss
        self.lambda_center = lambda_center
        self.num_classes = num_classes

        # To compute center loss
        self.centers = torch.zeros(num_classes, embeddings_dim)  # Initialize class centers

    def forward(self, embeddings, labels, mined_triplets=None):
        # Triplet loss
        if mined_triplets is not None:
            anchors, positives, negatives = mined_triplets
            triplet_loss = self.triplet_loss(embeddings[anchors], embeddings[positives], embeddings[negatives])
        else:
            triplet_loss = 0.0  # You may have a fallback or no triplet loss here

        # Center loss
        center_loss = self.center_loss(embeddings, labels)

        # Total loss
        total_loss = triplet_loss + self.lambda_center * center_loss
        return total_loss, triplet_loss, center_loss

    def center_loss(self, embeddings, labels):
        # Compute the class centers
        centers = self.centers[labels]  # Centers for the respective classes
        loss = torch.sum((embeddings - centers) ** 2)  # Euclidean distance between embeddings and centers
        return loss

    def update_centers(self, embeddings, labels, alpha=0.5):
        # Update the centers after each batch
        for i in range(self.num_classes):
            class_indices = torch.where(labels == i)[0]
            if len(class_indices) > 0:
                class_embeddings = embeddings[class_indices]
                self.centers[i] = alpha * self.centers[i] + (1 - alpha) * class_embeddings.mean(dim=0)