import matplotlib.pyplot as plt
import torch.optim as optim
import yaml
from logger import logging

from create_dataset import create_loader
from dataloader import load_data
from networks import *

with open("config.yaml", "r") as file:
    hyperparams = yaml.safe_load(file)

dataset = hyperparams["dataset"]
train_file_1 = hyperparams["train_file_1"]
train_file_2 = hyperparams["train_file_2"]  # can be None
test_file = hyperparams["test_file"]
indices_train_1 = (
    hyperparams["indices_train_1"]["from"],
    hyperparams["indices_train_1"]["to"],
)
indices_train_2 = (
    hyperparams["indices_train_2"]["from"],
    hyperparams["indices_train_2"]["to"],
)
indices_test_1 = (
    hyperparams["indices_test_1"]["from"],
    hyperparams["indices_test_1"]["to"],
)
indices_test_2 = (
    hyperparams["indices_test_2"]["from"],
    hyperparams["indices_test_2"]["to"],
)
if dataset == "alchemy":
    indices_train_1 = (0, 100)
    indices_train_2 = (100, 200)
    indices_test_1 = (0, 32)
    indices_test_2 = (32, 64)
    if train_file_1 is None:
        train_file_1 = "../Data/alchemy_train_wo_edgeattr_from_(0, 100)_to_(100, 200)"
    if test_file is None:
        test_file = "../Data/alchemy_test_wo_edgeattr_from_(0, 32)_to_(32, 64)"
if dataset == "aids":
    indices_train_1 = (0, 100)
    indices_train_2 = (0, 200)
    indices_test_1 = (0, 32)
    indices_test_2 = (32, 64)
    if train_file_1 is None:
        train_file_1 = "../Data/aids_train_wo_edgeattr_from_(0, 100)_to_(100, 200)"
    if train_file_2 is None:
        train_file_2 = "../Data/aids_train_wo_edgeattr_from_(0, 100)_to_(0, 100)"
    if test_file is None:
        test_file = "../Data/aids_test_wo_edgeattr_from_(0, 32)_to_(32, 64)"
model_dict = hyperparams["model"]
training_dict = hyperparams["training"]
num_epochs = training_dict["num_epochs"]
save_files = hyperparams["save_files"]
test_train = hyperparams["test_while_train"]
test_end = hyperparams["test_at_end"]
test_epochs = hyperparams["test_epochs"]
plot = hyperparams["plot"]
N_tests = hyperparams["N_tests"]
smooth_size = hyperparams["smooth_size"]
plot_nb = hyperparams["plot_nb"]

train_dataset, test_dataset = load_data(name=dataset)
num_node_features = len(train_dataset[0]["node_feat"][0])
train_dataset_1 = train_dataset[indices_train_1[0] : indices_train_1[1]]
train_dataset_2 = train_dataset[indices_train_2[0] : indices_train_2[1]]
test_dataset_1 = test_dataset[indices_test_1[0] : indices_test_1[1]]
test_dataset_2 = test_dataset[indices_test_2[0] : indices_test_2[1]]
train_geds = torch.load(train_file_1)
if train_file_2 is not None:
    train_geds_2 = torch.load(train_file_2)
    train_geds = torch.cat((train_geds, train_geds_2), dim=1)
num_nodes_train_1 = torch.tensor([graph["num_nodes"] for graph in train_dataset_1])
num_nodes_train_2 = torch.tensor([graph["num_nodes"] for graph in train_dataset_2])
node_sums_train = num_nodes_train_1.unsqueeze(1) + num_nodes_train_2.unsqueeze(0)
train_geds /= node_sums_train
train_geds = torch.exp(-0.5 * train_geds)
test_geds = torch.load(test_file)
num_nodes_test_1 = torch.tensor([graph["num_nodes"] for graph in test_dataset_1])
num_nodes_test_2 = torch.tensor([graph["num_nodes"] for graph in test_dataset_2])
node_sums_test = num_nodes_test_1.unsqueeze(1) + num_nodes_test_2.unsqueeze(0)
test_geds /= node_sums_test
test_geds = torch.exp(-0.5 * test_geds)

train_loader_1 = create_loader(
    train_dataset_1, batch_size=training_dict["batch_size"], shuffle=True
)
train_loader_2 = create_loader(
    train_dataset_2, batch_size=training_dict["batch_size"], shuffle=True
)
test_loader_1 = create_loader(
    test_dataset_1, batch_size=training_dict["batch_size"], shuffle=True
)
test_loader_2 = create_loader(
    test_dataset_2, batch_size=training_dict["batch_size"], shuffle=True
)

models = []
if model_dict["type"] == "GNN" or model_dict["type"] == "both":
    gnn = GNN(*eval(model_dict["gcn_channels"]), num_node_features=num_node_features)
    models.append(
        SiameseNetwork(
            gnn,
            model_dict["siamese_NTN"],
            model_dict["siamese_hidden_1"],
            model_dict["siamese_hidden_2"],
        )
    )
if model_dict["type"] == "FCN" or model_dict["type"] == "both":
    fcn = FCN(
        input_dim=num_node_features,
        hidden_dim=model_dict["fcn_hidden"],
        output_dim=model_dict["fcn_out"],
    )
    models.append(
        SiameseNetwork(
            fcn,
            model_dict["siamese_NTN"],
            model_dict["siamese_hidden_1"],
            model_dict["siamese_hidden_2"],
        )
    )


def test(model, test_loader_1, test_loader_2, loss_fn, test_geds, epochs=1):
    model.eval()
    total_loss = 0
    total_relative_error = 0
    for _ in range(epochs):
        with torch.no_grad():
            for batch1 in test_loader_1:
                for batch2 in test_loader_2:
                    geds = test_geds[batch1.idx, batch2.idx].unsqueeze(1)
                    predictions = model(batch1, batch2)
                    loss = loss_fn(predictions, geds)
                    total_loss += loss
                    relative_error = torch.mean(
                        torch.abs(predictions - geds)
                        / torch.max(torch.tensor(0.001), geds)
                    )
                    total_relative_error += relative_error
    total_loss /= len(test_loader_1) * len(test_loader_2) * epochs
    total_relative_error /= len(test_loader_1) * len(test_loader_2) * epochs
    return total_loss, total_relative_error


model_trajectories = {}
if test_train:
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

    if test_train:
        test_losses = test(
            model, test_loader_1, test_loader_2, loss_fn, test_geds, test_epochs
        )
        model_trajectories[model.type]["test_losses"].append(test_losses[0].item())
        model_trajectories[model.type]["test_accuracies"].append(test_losses[1].item())
        print(
            f"Before training, test loss: {test_losses[0]:.4f}, relative test error: {test_losses[1]:.4f}"
        )
    test_interval = max(1, num_epochs // N_tests)
    for epoch in range(num_epochs):
        model.train()
        for i, batch1 in enumerate(train_loader_1):
            for batch2 in train_loader_2:
                # extract ground truth GEDs from full GED matrix
                geds = train_geds[batch1.idx, batch2.idx].unsqueeze(1)
                predictions = model(batch1, batch2)
                loss = loss_fn(predictions, geds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (epoch + 1) % test_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: train loss {loss:.4f}")
            if test_train:
                test_losses = test(
                    model, test_loader_1, test_loader_2, loss_fn, test_geds, test_epochs
                )
                model_trajectories[model.type]["test_losses"].append(
                    test_losses[0].item()
                )
                model_trajectories[model.type]["test_accuracies"].append(
                    test_losses[1].item()
                )
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

    if test_end:
        test_losses = test(
            model, test_loader_1, test_loader_2, loss_fn, test_geds, test_epochs
        )
        print(
            f"Test performance after training: loss {test_losses[0]}; rel. error {test_losses[1]}"
        )


def smooth_curve(data_list: list, window_size: int = 50):
    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    weights = torch.ones(window_size) / window_size
    smoothed_tensor = torch.nn.functional.conv1d(
        data_tensor.view(1, 1, -1), weights.view(1, 1, -1)
    )
    smoothed_list = smoothed_tensor.squeeze().tolist()
    smoothed_list = data_list[: window_size // 2] + smoothed_list
    return smoothed_list


def plot_trajectories(trajectories, window_size=50, total_epochs=500, plot_nb=50):
    num_plots = len(trajectories)
    if num_plots > 1:
        fig, axs = plt.subplots(1, num_plots, figsize=(8, 8), sharey=True)
        for i, (model, trajectory) in enumerate(trajectories.items()):
            adjusted_plot_nb = min(plot_nb, len(trajectory["test_accuracies"]))
            adjusted_total_epochs = (
                total_epochs * adjusted_plot_nb / len(trajectory["test_accuracies"])
            )
            y_values = smooth_curve(
                trajectory["test_accuracies"][:adjusted_plot_nb],
                window_size=window_size,
            )
            nb_points = len(y_values)
            x_values = [
                i * (adjusted_total_epochs // nb_points) for i in range(nb_points)
            ]
            axs[i].plot(x_values, y_values)
            axs[i].set_title(model)
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel("Relative Error")
    else:
        adjusted_plot_nb = min(
            plot_nb, len(trajectories[model_dict["type"]]["test_accuracies"])
        )
        adjusted_total_epochs = (
            total_epochs
            * adjusted_plot_nb
            / len(trajectories[model_dict["type"]]["test_accuracies"])
        )
        y_values = smooth_curve(
            trajectories[model_dict["type"]]["test_accuracies"][:adjusted_plot_nb],
            window_size=window_size,
        )
        nb_points = len(y_values)
        x_values = [i * (adjusted_total_epochs // nb_points) for i in range(nb_points)]
        plt.plot(x_values, y_values)
        plt.title(model_dict["type"])
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
    plt.tight_layout()
    plt.show()


if plot:
    plot_trajectories(model_trajectories, smooth_size, num_epochs, plot_nb)
