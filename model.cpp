#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

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


template <typename DataLoader>
void train(
    size_t epoch,
    VGG16Impl& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    VGG16Impl& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

auto ReadNormalizedDataSet(torch::data::datasets::MNIST::Mode dataset)
{
    auto readDataset = torch::data::datasets::MNIST(kDataRoot, dataset);
    auto normalizedDataset = readDataset.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
    return normalizedDataset;
}

auto main() -> int {
  torch::manual_seed(1);
  torch::Device device(torch::kCPU);

  VGG16Impl model(10);
  model.to(device);

  auto train_dataset = ReadNormalizedDataSet( torch::data::datasets::MNIST::Mode::kTrain);
  const size_t train_dataset_size = train_dataset.size().value();

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = ReadNormalizedDataSet( torch::data::datasets::MNIST::Mode::kTest);
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
  torch::serialize::OutputArchive archive;
  model.save(archive);
  archive.save_to("model_weights.pt");
}