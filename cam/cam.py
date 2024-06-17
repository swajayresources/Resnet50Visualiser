import torch
import torch.nn.functional as F




from cam import BaseCAM

# class CAM(BaseCAM):
#     def __init__(self, model, target_layer=None):
#         super().__init__(model, target_layer)
#         self.weight_softmax = torch.squeeze(self.get_weight(model))
#
#     def get_weight(self, model):
#         params = list(model.parameters())
#         return params[-2]
#
#     def forward(self, x, class_idx=None, retain_graph=False):
#         x.requires_grad = False
#         b, c, h, w = x.size()
#
#         # predication on raw x
#         logit = F.softmax(self.model(x), dim=1)
#
#         if class_idx is None:
#             score = logit[:, logit.max(1)[-1]].squeeze()
#         else:
#             score = logit[:, class_idx].squeeze()
#
#         activations = self.activations['value'].data
#         b, k, u, v = activations.size()
#
#         weights = self.weight_softmax[class_idx].view(b, k, 1, 1).data
#         saliency_map = (weights * activations).sum(1, keepdim=True)
#
#         saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
#         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#         saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
#         return saliency_map.detach(), logit.detach()
#
#     def __call__(self, x, class_idx=None, retain_graph=False):
#         return self.forward(x, class_idx, retain_graph)

class CAM(BaseCAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer)
        self.weight_softmax = torch.squeeze(self.get_weight(model))

    """__init__ method: Initializes the CAM object.
Calls the BaseCAM constructor: Sets up the model and hooks.
Retrieves and stores the model's weights: Uses get_weight to get the second-to-last layer's weights and squeezes the tensor to remove dimensions of size 1."""

    def get_weight(self, model):
        params = list(model.parameters())
        return params[-2]
    
    """get_weight method: Extracts the second-to-last parameter (typically the final linear layer's weights) from the model."""

    def forward(self, x, class_idx=None, retain_graph=False):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
            x = x.to(next(self.model.parameters()).device)

        # x.requires_grad = False
        b, c, h, w = x.size()

        # predication on raw x
        logit = F.softmax(self.model(x), dim=1)

        if class_idx is None:
            class_idx = logit.max(1)[-1]

        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        weights = self.weight_softmax[class_idx].view(1, k, 1, 1).data
        saliency_map = (weights * activations).sum(1, keepdim=True)

        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_shape = saliency_map.shape
        saliency_map = saliency_map.view(saliency_map.shape[0], -1)
        saliency_map_min, saliency_map_max = saliency_map.min(1, keepdim=True)[0], saliency_map.max(1, keepdim=True)[0]
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        saliency_map = saliency_map.view(saliency_map_shape)
        return saliency_map.detach().cpu().numpy(), logit.detach()
    

    """forward method: Computes the CAM for a given input x.
Ensures x is 4D: Adds a batch dimension if necessary and moves x to the correct device.
Computes the logit (softmax) values: Passes x through the model.
Determines class_idx: If not provided, uses the class with the highest logit value.
Retrieves activations: Gets the activations captured by the forward hook.
Calculates the saliency map: Multiplies activations by weights corresponding to class_idx and sums along the channel dimension.
Resizes the saliency map: Uses bilinear interpolation to match the input size.
Normalizes the saliency map: Scales the saliency map values to the range [0, 1].
Returns the saliency map and logits: As numpy arrays and PyTorch tensors, respectively."""

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
    
    """__call__ method: Makes the CAM object callable, passing arguments to forward"""