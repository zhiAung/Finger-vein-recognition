from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

    # test_loader=DataLoader(test_dataset,batch_size=1)


img, label = iter(training_data_loader).next()
# print(img.numpy()[0,:,:,:].shape)
plt.imshow(img.numpy()[0, :, :, :].transpose((1, 2, 0)), cmap='gray')  # 将原来的tensor[1,3,227,227]数据转换为array[227,227,3]
plt.show()
img, label = Variable(img, volatile=True), Variable(label, volatile=True)

myresnet = models.resnet152(pretrained=True)
exact_list = ["layer1", "layer2", "layer3", 'layer4']
myexactor = FeatureExtractor(myresnet, exact_list)
x = myexactor(img)
# print(x)
# plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='gray')
# plt.imshow(x[1].data.numpy()[0,i,:,:],cmap='gray')
# plt.show()
print("layer1")
print(x[0].data.numpy().shape)
for i in range(10):
    '''ax = plt.subplot(8, 8, i + 1)  
    ax.set_title('Sample #{}'.format(i))  
    ax.axis('off')  '''
    plt.imshow(x[0].data.numpy()[0, i, :, :], cmap='gray')
    plt.show()
print("layer2")
print(x[1].data.numpy().shape)
for i in range(10):
    '''ax = plt.subplot(8, 8, i + 1)  
    ax.set_title('Sample #{}'.format(i))  
    ax.axis('off')  '''

    plt.imshow(x[1].data.numpy()[0, i, :, :], cmap='gray')
    plt.show()
print("layer3")
print(x[2].data.numpy().shape)
for i in range(10):
    '''ax = plt.subplot(8, 8, i + 1)  
    ax.set_title('Sample #{}'.format(i))  
    ax.axis('off')  '''

    plt.imshow(x[2].data.numpy()[0, i, :, :], cmap='gray')
    plt.show()
print("layer4")
print(x[3].data.numpy().shape)
for i in range(10):
    '''ax = plt.subplot(8, 8, i + 1)  
    ax.set_title('Sample #{}'.format(i))  
    ax.axis('off')  '''

    plt.imshow(x[3].data.numpy()[0, i, :, :], cmap='gray')
    plt.show()