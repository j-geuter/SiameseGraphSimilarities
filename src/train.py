import yaml
from logger import logging
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from create_dataset import create_loader
from dataloader import load_data
from networks import *

with open("config.yaml", "r") as file:
    hyperparams = yaml.safe_load(file)

train_file = hyperparams["train_file"]
test_file = hyperparams["test_file"]
N_train = hyperparams["N_train"]
N_test = hyperparams["N_test"]
model_dict = hyperparams["model"]
training_dict = hyperparams["training"]
num_epochs = training_dict["num_epochs"]
save_files = hyperparams["save_files"]

train_dataset, test_dataset = load_data()
train_dataset = train_dataset[:N_train]
test_dataset = test_dataset[:N_test]
train_geds = torch.load(train_file)
num_nodes_train = torch.tensor([graph['num_nodes'] for graph in train_dataset])
node_sums_train = num_nodes_train.unsqueeze(1) + num_nodes_train.unsqueeze(0)
train_geds /= node_sums_train
train_geds = torch.exp(- train_geds)
test_geds = torch.load(test_file)
num_nodes_test = torch.tensor([graph['num_nodes'] for graph in test_dataset])
node_sums_test = num_nodes_test.unsqueeze(1) + num_nodes_test.unsqueeze(0)
test_geds /= node_sums_test
test_geds = torch.exp(- test_geds)

train_loader = create_loader(
    train_dataset, batch_size=training_dict["batch_size"], shuffle=True
)
test_loader = create_loader(
    test_dataset, batch_size=training_dict["batch_size"], shuffle=False
)

models = []
if model_dict["type"] == "GNN" or model_dict["type"] == "both":
    gnn = GNN(hidden_channels=model_dict["gcn_channels"])
    models.append(SiameseNetwork(gnn))
if model_dict["type"] == "FCN" or model_dict["type"] == "both":
    fcn = FCN(hidden_dim=model_dict["fcn_hidden"], output_dim=model_dict["fcn_out"])
    models.append(SiameseNetwork(fcn))


def test(model, test_loader, loss_fn, test_geds):
    model.eval()
    total_loss = 0
    total_relative_error = 0
    with torch.no_grad():
        for batch1, batch2 in zip(test_loader, test_loader):
            geds = test_geds[batch1.idx, batch2.idx].unsqueeze(1)
            predictions = model(batch1, batch2)
            loss = loss_fn(predictions, geds)
            total_loss += loss
            relative_error = torch.mean(torch.abs(predictions - geds) / torch.max(torch.tensor(1), geds))
            total_relative_error += relative_error
    total_loss /= len(test_loader)
    total_relative_error /= len(test_loader)
    return total_loss, total_relative_error


model_trajectories = {}
for model in models:
    model_trajectories[model.type] = {"test_losses": [], "test_accuracies": []}

for model in models:
    logging.info(f"Training model {model.type}")
    if training_dict["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=training_dict["learn_rate"])
    else:
        raise ValueError(f'Optimizer {training_dict["optimizer"]} not implemented.')
    if training_dict["loss"] == "MSE":
        loss_fn = nn.MSELoss()
    elif training_dict["loss"] == "L1":
        loss_fn = nn.L1Loss()
    else:
        raise ValueError(f'Loss {training_dict["loss"]} not implemented.')

    test_losses = test(model, test_loader, loss_fn, test_geds)
    model_trajectories[model.type]["test_losses"].append(test_losses[0].item())
    model_trajectories[model.type]["test_accuracies"].append(test_losses[1].item())
    print(
        f"Before training, test loss: {test_losses[0]:.4f}, relative test error: {test_losses[1]:.4f}"
    )

    for epoch in range(num_epochs):
        model.train()
        for i, (batch1, batch2) in enumerate(zip(train_loader, train_loader)):

            # extract ground truth GEDs from full GED matrix
            geds = train_geds[batch1.idx, batch2.idx].unsqueeze(1)
            predictions = model(batch1, batch2)
            loss = loss_fn(predictions, geds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (100*epoch/num_epochs)%10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: train loss {loss:.4f}")
        test_losses = test(model, test_loader, loss_fn, test_geds)
        model_trajectories[model.type]["test_losses"].append(test_losses[0].item())
        model_trajectories[model.type]["test_accuracies"].append(test_losses[1].item())
        if (100 * epoch / num_epochs) % 10 == 0:
            print(
                f"Epoch {epoch+1}, test loss: {test_losses[0]:.4f}, relative test error: {test_losses[1]:.4f}"
            )

    file_name = save_files[model.type]
    torch.save(
        {
            "params": hyperparams,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "trajectory": model_trajectories[model.type],
        },
        file_name,
    )

logging.info("Training finished. Plotting models:")


def smooth_curve(data_list: list, window_size: int = 10):
    first_entry = data_list[0]
    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    weights = torch.ones(window_size) / window_size
    smoothed_tensor = torch.nn.functional.conv1d(data_tensor.view(1, 1, -1), weights.view(1, 1, -1))
    smoothed_list = smoothed_tensor.squeeze().tolist()
    smoothed_list[0] = first_entry
    return smoothed_list


for key, value in model_trajectories.items():
    value['test_accuracies'] = smooth_curve(value['test_accuracies'])


def plot_trajectories(trajectories):
    num_plots = len(trajectories)
    fig, axs = plt.subplots(1, num_plots, figsize=(8, 8))
    for i, (model, trajectory) in enumerate(trajectories.items()):
        axs[i].plot(trajectory["test_accuracies"])
        axs[i].set_title(model)
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Relative Error")
    plt.tight_layout()
    plt.show()


plot_trajectories(model_trajectories)
