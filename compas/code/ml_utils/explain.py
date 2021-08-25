# TODO: correctly attribute Christopher J Anders and Grégoire Montavon
import copy
import torch
Module = torch.nn.modules.Module


def stabilize(z):
    return z + ((z == 0.).to(z) + z.sign()) * 1e-6


def newlayer(layer, g):
    layer = copy.deepcopy(layer)

    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass

    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass

    return layer


def collect_leaves(module):
    result = []
    children = list(module.children())
    if children:
        for child in children:
            result += collect_leaves(child)
    else:
        result.append(module)
    return result


class Conv(Module):
    def __init__(self, conv):
        Module.__init__(self)
        self.conv = conv


class ConvP(Conv):
    def __init__(self, conv, freeze_b=False, flatten=False):
        Conv.__init__(self, conv)
        self.pconv = newlayer(conv, lambda p: p.clamp(min=0))
        self.nconv = newlayer(conv, lambda p: p.clamp(max=0))
        self.div = None
        self.flatten = flatten
        if self.pconv.bias is not None:
            self.pconv.bias.requires_grad_(not freeze_b)
            self.nconv.bias.requires_grad_(not freeze_b)

    def forward(self, X):
        X, L, H = X
        if self.flatten:
            X = torch.nn.Flatten()(X)
            L = torch.nn.Flatten()(L)
            H = torch.nn.Flatten()(H)
        z = self.conv.forward(X)

        zp = z - self.pconv.forward(L) - self.nconv.forward(H)
        zp = stabilize(zp)
        self.div = (z / zp)
        return zp * self.div.data


class ConvI(Conv):
    def __init__(self, conv, rho, incr, freeze_b=False, flatten=False):
        Conv.__init__(self, conv)
        self.mconv = newlayer(conv, rho)
        self.incr = incr
        self.div = None
        self.flatten = flatten
        if self.conv.bias is None:
            self.mconv.bias = None
        else:
            self.mconv.bias.requires_grad_(not freeze_b)

    def forward(self, X):
        # Reduce X L H input
        if type(X) in [tuple, list]:
            X, _, _ = X
        if self.flatten:
            X = torch.nn.Flatten()(X)
        z = self.conv.forward(X)
        zp = self.mconv.forward(X)
        zp = self.incr(zp)
        self.div = (z / zp)
        return zp * self.div.data


def is_weight_layer(module):
    return isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d) or isinstance(module, torch.nn.Linear)


def get_lrp_rule(layers, rule="2"):
    """Returns a configuration that assigns LRP-y/-e/-0 to the first, second, and third third of the weight layers,
    respectively."""
    n_weight_layers = 0
    for layer in layers:
        if is_weight_layer(layer):
            n_weight_layers += 1

    lrp_layers = []

    if rule == "0":
        lrp_types = ["0"]
    elif rule == "e":
        lrp_types = ["e"]
    elif rule == "1":
        lrp_types = ["y"]
    elif rule == "1e":
        lrp_types = ["ye"]
    elif rule == "2":
        lrp_types = ["y", "e"]
    elif rule == "2e":
        lrp_types = ["ye", "e"]
    elif rule == "3":
        lrp_types = ["y", "e", "0"]
    else:
        raise ValueError("Illegal LRP rule: " + str(rule))

    divisor = len(lrp_types)
    for t_i, t in enumerate(lrp_types):
        n = (n_weight_layers + (divisor-1-t_i)) // (divisor-t_i)
        lrp_layers += [t] * n
        n_weight_layers -= n
    return lrp_layers


def make_explainable(module_seq_, gamma=0.25, lrp_rule="e"):
    module_seq = copy.deepcopy(module_seq_)
    do_flatten = False
    n_modules = len(list(module_seq))
    lrp_layers = get_lrp_rule(module_seq, lrp_rule)
    weight_layer_cnt = 0
    for i in range(n_modules):
        if isinstance(module_seq[i], torch.nn.Flatten):
            module_seq[i] = torch.nn.Sequential()
            do_flatten = True
        elif is_weight_layer(module_seq[i]):
            if lrp_layers[weight_layer_cnt] == "p":
                module_seq[i] = ConvP(module_seq[i], flatten=do_flatten, freeze_b=True)
            else:
                if "y" in lrp_layers[weight_layer_cnt]:
                    def rho(p):
                        return p + gamma * p.clamp(min=0)
                else:
                    def rho(p):
                        return p
                if "e" in lrp_layers[weight_layer_cnt]:
                    def incr(z):
                        return stabilize(z)
                else:
                    def incr(z):
                        return z
                module_seq[i] = ConvI(module_seq[i], rho, incr, flatten=do_flatten, freeze_b=True)
            do_flatten = False
            weight_layer_cnt += 1

    return module_seq


class ApplyToList(Module):
    def __init__(self, module_to_apply):
        Module.__init__(self)
        self.mod = module_to_apply

    def forward(self, x):
        if type(x) in [list, tuple]:
            return [self.mod(elem) for elem in x]
        else:
            return self.mod(x)


def grad(model, X, T=None, times_input=True):
    # Require that gradients of the input are computed
    X.requires_grad_(True)
    X.grad = None

    # Perform gradient x input
    Y = model(X)
    explanation_domain = X

    # Select output to explain if Y is multi-dimensional
    if T is not None:
        T.requires_grad_(False)
        if len(T.shape) == 1 and int(T.shape[0]) == 1:
            # T is scalar
            to_explain = Y[0, T].sum()
        else:
            # T is vector
            to_explain = (Y * T).sum()
    else:
        to_explain = Y

    grads = torch.autograd.grad(to_explain, explanation_domain)

    if times_input:
        R = grads[0] * explanation_domain
    else:
        R = grads[0]

    return R
