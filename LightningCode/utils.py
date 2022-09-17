from torch.autograd import Function


class ConstMultiplication(Function):
    @staticmethod
    def forward(ctx, input, const):
        ctx.constant = const
        return input*const

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.constant, None
