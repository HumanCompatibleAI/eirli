"""General Python and PyTorch utilities."""


def freeze_params(module):
    """Modifies Torch module in-place to convert all its parameters to buffers,
    and give them require_grad=False. This is a slightly hacky way of
    "freezing" the module."""

    # We maintain this stack so that we can traverse the module tree
    # depth-first. We'll terminate once we've traversed all modules.
    module_stack = [module]

    while module_stack:
        # get next module from end of the stack
        next_module = module_stack.pop()

        # sanity check to ensure we only have named params
        param_list = list(next_module.parameters(recurse=False))
        named_param_list = list(next_module.named_parameters(recurse=False))
        assert len(param_list) == len(named_param_list), \
            f"cannot handle module '{next_module}' with unnamed parameters"

        # now remove each param (delattr) and replace it with a buffer
        # (register_buffer)
        for param_name, param_var in named_param_list:
            param_tensor = param_var.data.clone().detach()
            assert not param_tensor.requires_grad
            delattr(next_module, param_name)
            next_module.register_buffer(param_name, param_tensor)

        # do the same for child modules
        module_stack.extend(next_module.children())

    # sanity check to make sure we have no params on the root module
    remaining_params = list(module.parameters())
    assert len(remaining_params) == 0, \
        f"module '{module}' has params remaining: {remaining_params}"
