import torch
import torch.nn.functional as F
from cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)

    """__init__ method: Initializes the GradCAM object by calling the BaseCAM constructor, setting up the model and target layer for capturing activations and gradients."""

    def forward(self, x, class_idx=None, retain_graph=False): ##nitialization Method: This method initializes the GradCAM object, calling the BaseCAM constructor. It sets up the model and target layer to capture activations and gradients.
        if len(x.size()) == 3:
            x = x.unsqueeze(0)#Ensures x is 4D: Adds a batch dimension if necessary.

        x = x.to(next(self.model.parameters()).device)
        b, c, h, w = x.size()

        # predication on raw x
        logit = self.model(x)
        softmax = F.softmax(logit, dim=1)
        """Model Prediction: Computes the raw logits by passing x through the model and applies softmax to get class probabilities."""

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]]#.squeeze()
        else:
            score = logit[:, class_idx]#.squeeze()
            # score = logit[:, class_idx]


        """Class Score Selection: Determines the score for the class of interest. If class_idx is not provided, it uses the class with the highest score."""

        if b > 1:
            retain_graph = True
            """Graph Retention: Ensures the computation graph is retained for multi-sample batches to allow multiple backward passes.
python"""

        self.model.zero_grad()
        gradients_list = []
        for i, item in enumerate(score):
            item.backward(retain_graph=retain_graph)
            gradients = self.gradients['value'].data[i]
            gradients_list.append(gradients)

        """Gradient Computation: Computes gradients of the class score w.r.t. feature maps, iterating over each item in the batch and storing gradients."""

        gradients = torch.stack(gradients_list, dim=0)
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        """Stacking Gradients: Stacks gradients for each sample in the batch and retrieves the activations from the target layer."""

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)


        """Weight Calculation: Computes weights by averaging gradients over spatial dimensions and applies them to the activations to create the saliency map."""

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        """ReLU and Upsampling: Applies ReLU to the saliency map and upsamples it to the original input size using bilinear interpolation."""
        # saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        # saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

        saliency_map_shape = saliency_map.shape
        saliency_map = saliency_map.view(saliency_map.shape[0], -1)
        saliency_map_min, saliency_map_max = saliency_map.min(1, keepdim=True)[0], saliency_map.max(1, keepdim=True)[0]
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        saliency_map = saliency_map.view(saliency_map_shape)
        """Normalization: Flattens the saliency map for normalization, ensuring values are between 0 and 1, then reshapes it back to the original format."""

        # import cv2
        # import numpy as np
        # map = saliency_map.cpu().data
        # map = cv2.applyColorMap(np.uint8(255 * map.squeeze()), cv2.COLORMAP_JET)
        # cv2.imwrite('test.jpg', map)

        return saliency_map.detach().cpu().numpy(), softmax.detach()
    
    """Return Statement: Returns the normalized saliency map and softmax output, detaching them from the computation graph and moving them to CPU for further processing."""

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
    """Callable Method: This method makes the object callable like a function, directly invoking the forward method with the provided arguments. It simplifies the usage of the class by allowing an instance to be called with input data, class index, and retain_graph flag."""


class GradCAMpp(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"): 
        super().__init__(model, target_layer)

    """Initialization Method: Initializes the GradCAMpp object by calling the BaseCAM constructor. This sets up the model and target layer to capture activations and gradients."""

    def forward(self, x, class_idx=None, retain_graph=False):
        b, c, h, w = x.size()

        # predication on raw x
        logit = self.model(x)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = activations.size()

        alpha_num = gradients.pow(2)#Alpha Numerator: Computes the numerator for alpha, which is the squared gradients.
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        #Alpha Denominator: Computes the denominator for alpha, a combination of squared gradients and the product of activations and cubed gradients.
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)
        #Alpha Calculation: Ensures non-zero denominator and computes alpha by dividing the numerator by the denominator, adding a small epsilon for numerical stability.
        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)
        #Weight Calculation: Computes positive gradients and weights by applying alpha, reshaping, and summing over spatial dimensions.

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        #Saliency Map: Applies weights to activations to create the saliency map and applies ReLU to keep only positive values.
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min).data
        #Upsampling and Normalization: Upsamples the saliency map to the original input size and normalizes it to ensure values are between 0 and 1.

        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
    #Callable Method: Allows the object to be called like a function, invoking the forward method with the provided arguments. This simplifies the usage of the class.


class SmoothGradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2", stdev_spread=0.15, n_samples=20, magnitude=True):
        super().__init__(model, target_layer)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        """Initialization Method: Initializes the SmoothGradCAM object by calling the BaseCAM constructor. It sets up the model, target layer, standard deviation spread, number of samples, and magnitude flag.
stdev_spread: Controls the amount of noise added to the input.
n_samples: Number of noisy samples to generate.
magnitude: Whether to use the magnitude of gradients."""

    def forward(self, x, class_idx=None, retain_graph=False):
        b, c, h, w = x.size()

        if class_idx is None:
            predicted_class = self.model(x).max(1)[-1]
        else:
            predicted_class = torch.LongTensor([class_idx])

        saliency_map = 0.0

        stdev = self.stdev_spread / (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev
        """Standard Deviation Calculation: Computes the standard deviation for the noise based on the spread and input range, creating a tensor of the same shape as x."""

        self.model.zero_grad()
        for i in range(self.n_samples):
            x_plus_noise = torch.normal(mean=x, std=std_tensor)
            x_plus_noise.requires_grad_()
            x_plus_noise.cuda()
            """Noisy Sample Generation: Iterates over the number of samples, generating noisy versions of the input. Each noisy sample requires gradient computation and is moved to the GPU."""
            logit = self.model(x_plus_noise)
            score = logit[0][predicted_class]
            score.backward(retain_graph=True)
            """Model Prediction and Backward Pass: Computes logits for the noisy sample, retrieves the score for the predicted class, and performs a backward pass to compute gradients."""

            gradients = self.gradients['value']
            if self.magnitude:
                gradients = gradients * gradients
            activations = self.activations['value']
            b, k, u, v = activations.size()
            """Gradient and Activation Retrieval: Retrieves the gradients and activations for the target layer. If magnitude is true, it squares the gradients."""

            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            """Weight Calculation: Computes the weights by averaging gradients over the spatial dimensions and reshaping them."""

            saliency_map += (weights * activations).sum(1, keepdim=True).data
            """Accumulate Saliency Map: Adds the weighted activations to the saliency map, accumulating contributions from all noisy samples."""

        saliency_map = F.relu(saliency_map)
        """ReLU Activation: Applies ReLU to the saliency map to keep only positive values."""
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min).data
        """Upsampling and Normalization: Upsamples the saliency map to the original input size and normalizes it to ensure values are between 0 and 1."""

        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)