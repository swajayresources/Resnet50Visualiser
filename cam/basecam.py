class BaseCAM(object):
    def __init__(self, model, target_layer="module.layer4.2"):
        super(BaseCAM, self).__init__()
        self.model = model.eval()
        self.gradients = dict()
        self.activations = dict()


        """__init__ method: Initializes the BaseCAM object.
Calls the superclass's __init__ method: Ensures proper initialization of any inherited attributes.
Sets the model to evaluation mode (model.eval()): Ensures the model behaves correctly during inference.
Initializes self.gradients and self.activations: These dictionaries will store gradients and activations of the target layer."""

        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)


                """Iterates through the model's named modules: Identifies the module with the name matching target_layer.
Registers hooks: Attaches forward_hook and backward_hook to the target layer to capture activations and gradients."""

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]
    

    """backward_hook method: Captures the gradients during the backward pass.
Stores the gradients in self.gradients: Specifically, the gradient of the output of the target layer."""

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    """forward_hook method: Captures the activations during the forward pass.
Stores the activations in self.activations: Specifically, the output of the target layer."""

    def forward(self, x, class_idx=None, retain_graph=False):
        raise NotImplementedError
    
    """forward method: Intended to perform the forward pass and compute CAM. It must be implemented in subclasses.
Raises NotImplementedError: Indicates that this method needs to be overridden in a subclass."""

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
    
    """__call__ method: Makes an instance of BaseCAM callable.
Calls the forward method: Passes the arguments to forward."""