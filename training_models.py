
from cnn.cnn import *


def train_model_and_save(
        train_loader,
        valid_loader,
        model,
        optimizer,
        criterion,
        num_epochs,
        name_saving_model,
        device,
        lr,
        previous_epochs=0,
):
    history = model.train_model(
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        lr=lr,
        epochs=num_epochs,
        device=device,
        previous_epochs=previous_epochs,
    )

    model.save(name_saving_model)
    return history


def load_previous_model(model, name_saving_model):
    model_weights = load_model_weights(name_saving_model)
    my_trained_model = model
    my_trained_model.load_state_dict(model_weights)
    return my_trained_model


def load_training_data(batch_size):
    train_dir = "./dataset/training"
    valid_dir = "./dataset/validation"

    train_loader, valid_loader, num_classes = load_data(
        train_dir, valid_dir, batch_size=batch_size, img_size=224
    )
    return train_loader, valid_loader, num_classes


def main_train(batch_size, dropout, lr, unfreezed_layers, new_epochs, curr_epochs, device, retrain=False,
               model_name=""):
    train_loader, valid_loader, num_classes = load_training_data(batch_size)

    model = CNN(torchvision.models.resnet101(weights='DEFAULT'), num_classes, unfreezed_layers=int(unfreezed_layers),
                dropout=float(dropout)).to(device)

    if retrain:
        model = load_previous_model(model, model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss()

    name_saving_model = f"resnet101-{new_epochs + curr_epochs}epoch-{lr}lr-{batch_size}bs-{dropout}do-{unfreezed_layers}unlay"
    history = train_model_and_save(train_loader,
                                   valid_loader,
                                   model,
                                   optimizer,
                                   criterion,
                                   new_epochs,
                                   name_saving_model,
                                   device,
                                   lr,
                                   previous_epochs=curr_epochs,
                                   )
    return history


if __name__ == "__main__":
    wandb.login(key="50f7eb3c5f21aecac69f018cf7ec6bcaecce16cc")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs, dp, lr, ul = 128, 0.2, 0.0001, 3
    history = main_train(bs, dp, lr, ul, 3, 20, device, model_name="resnet101-20epoch-0.0001lr-128bs-0.2do-3unlay",
                         retrain=True)