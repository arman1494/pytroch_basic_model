/*
The implementation of VGG16 network includes 13 Conv2d and 3 Dense layers.
*/
class VGG16Impl : public torch::nn::Module {
public:
    VGG16Impl(int num_classes)
        :/*
            Firts Block:
                Conv2d -> BatchNormal2d -> ReLU -> Conv2d -> Conv2d -> BatchNormal2d -> ReLU -> MaxPool2
        */ 
        block_1(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, /*kernel_size=*/3).stride(/*stride=*/1).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(/*kernel_size=*/1)
            )),

        /*
            Second Block:
                Conv2d -> BatchNormal2d -> ReLU -> Conv2d -> Conv2d -> BatchNormal2d -> ReLU -> MaxPool2
        */
        block_2(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, /*kernel_size=*/3).stride(/*stride=*/1).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(/*kernel_size=*/1)
        )),
        /*
            Third Block:
                Conv2d -> BatchNormal2d -> ReLU -> Conv2d -> Conv2d -> BatchNormal2d -> ReLU -> MaxPool2
        */
        block_3(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, /*kernel_size=*/3).stride(/*stride=*/1).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(/*kernel_size=*/1)
        )),
        /*
            Fourth Block
                Conv2d -> BatchNormal2d -> ReLU -> Conv2d -> Conv2d -> BatchNormal2d -> ReLU -> MaxPool2
        */
        block_4(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, /*kernel_size=*/3).stride(/*stride=*/1).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(/*kernel_size=*/1)
        )),
        /*
            Fifth Block:
                Conv2d -> BatchNormal2d -> ReLU -> Conv2d -> Conv2d -> BatchNormal2d -> ReLU -> MaxPool2
        */
        block_5(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).stride(/*stride=*/1).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(/*padding=*/1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(/*kernel_size=*/1)
        )),
        /*
            classifier
        */
        classifier(torch::nn::Sequential(
            torch::nn::Linear(512, 4096),
            torch::nn::ReLU(),
             torch::nn::Dropout(torch::nn::DropoutOptions().p(0.25)),
            torch::nn::Linear(4096, 50176),
            torch::nn::ReLU(),
             torch::nn::Dropout(torch::nn::DropoutOptions().p(0.25)),
            torch::nn::Linear(50176, num_classes)
        ))

    {
        register_module("block_1", block_1);
        register_module("block_2", block_2);
        register_module("block_3", block_3);
        register_module("block_4", block_4);
        register_module("block_5", block_5);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = block_1->forward(x);
        x = block_2->forward(x);
        x = block_3->forward(x);
        x = block_4->forward(x);
        x = block_5->forward(x);
        x = x.view({-1, 512});
        x = classifier->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
        
    }

private:
    torch::nn::Sequential block_1;
    torch::nn::Sequential block_2;
    torch::nn::Sequential block_3;
    torch::nn::Sequential block_4;
    torch::nn::Sequential block_5;
    torch::nn::Sequential classifier;
};


class Net : public torch::nn::Module {
    private:
        torch::nn::Sequential block_1;
        torch::nn::Sequential block_2;
        torch::nn::Sequential block_3;
    public:
        Net():
            block_1(torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
                torch::nn::MaxPool2d(/*kernel_size=*/2),
                torch::nn::ReLU())),
            block_2(torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
                torch::nn::MaxPool2d(/*kernel_size=*/2),
                torch::nn::ReLU())),
            block_3(torch::nn::Sequential( torch::nn::Linear(320, 50),
            torch::nn::ReLU(),
            torch::nn::Dropout(torch::nn::DropoutOptions().p(0.25)),
            torch::nn::Linear(50, 10)))
            {
                register_module("block_1", block_1);
                register_module("block_2", block_2);
                register_module("block_3", block_3);
            }
        torch::Tensor forward(torch::Tensor x) {
            x = block_1->forward(x);
            x = block_2->forward(x);
            x = x.view({-1, 320});
            x = block_3->forward(x);
            return torch::log_softmax(x, /*dim=*/1);
        }

};