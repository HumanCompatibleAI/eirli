from collections import OrderedDict
import textwrap

import numpy as np
import torch
import torch.nn as nn


def get_device(device):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    if isinstance(device, str):
        device = torch.device(device)
    return device


def summary(*args, **kwargs):
    result, params_info = summary_string(*args, **kwargs)
    print(result)
    return params_info


def compute_shape(network_object, batch_size):
    if isinstance(network_object, (list, tuple)) \
       and all(hasattr(t, 'size') for t in network_object):
        return [
            [-1] + list(o.size())[1:] for o in network_object
        ]
    elif hasattr(network_object, 'size'):
        out_shape = list(network_object.size())
        out_shape[0] = batch_size
        return out_shape
    # unknown network_object shape
    return None


def summary_string(model, input_size, batch_size=-1, device=None, dtypes=None,
                   input_range=(0, 1.0)):
    if dtypes is None:
        dtypes = [torch.FloatTensor]*len(input_size)

    device = get_device(device)

    summary_str = ''

    def register_hook(module):
        def hook(module, input_tens, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] \
                = compute_shape(input_tens, batch_size)
            summary[m_key]["output_shape"] = compute_shape(output, batch_size)

            nb_params = 0
            trainable = False
            for param_tensor in module.parameters(recurse=False):
                trainable = trainable or param_tensor.requires_grad
                param_shape = torch.LongTensor(list(param_tensor.size()))
                nb_params += torch.prod(param_shape)
            summary[m_key]["trainable"] = trainable
            summary[m_key]["nb_params"] = nb_params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    lo, hi = input_range
    x = [torch.rand(2, *in_size).type(dtype).to(device=device) * (hi - lo) + lo
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    head_width = 64
    summary_str += "-" * head_width + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "=" * head_width + "\n"
    total_params = 0
    total_output = 0
    unknown_shapes = False
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        out_shape = summary[layer]["output_shape"]
        if out_shape is not None:
            if isinstance(out_shape[0], list):
                # this happens if a layer has multiple outputs
                # (we just add up the individual sizes)
                sizes = out_shape
            else:
                sizes = [out_shape]
            for size in sizes:
                # skip over the batch axes (-1)
                total_output += np.prod([s for s in size if s != -1])
            shape_line = str(summary[layer]["output_shape"])
        else:
            unknown_shapes = True
            shape_line = "???"

        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            shape_line,
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] is True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * head_width + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}" \
                   .format(total_params - trainable_params) + "\n"
    summary_str += "-" * head_width + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" \
                   % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * head_width + "\n"
    if unknown_shapes:
        summary_str += "\n".join(textwrap.wrap(
            "WARNING: Could not determine output shapes for some layers, "
            "forward/backward pass size may be higher than listed above.",
            width=head_width))
    # return summary
    return summary_str, (total_params, trainable_params)
