import torch
import torchvision


class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68]).resize_(1, 3, 1, 1))
        self.resnet = torch.jit.trace(torchvision.models.resnet18(pretrained=True), torch.rand(1, 3, 224, 224))
        # self.resnet = torchvision.models.resnet18(pretrained=True)

    def forward(self, input):
        return self.resnet(input - self.means)


if __name__ == '__main__':
    # resnet = torchvision.models.resnet18()
    # # traced_resnet = torch.jit.trace(resnet, torch.rand(1, 3, 224, 224))
    # script_module = torch.jit.script(resnet)
    # script_module_gpu = torch.jit.script(resnet.cuda())
    #
    # print(script_module.graph)
    module = MyScriptModule()
    # x = torch.randn((8, 3, 224, 224))
    script_module = torch.jit.script(module)
    print(script_module.code)
    torch.jit.save(script_module, "script_module.pt")
    loaded = torch.jit.load("script_module.pt")
    a=1