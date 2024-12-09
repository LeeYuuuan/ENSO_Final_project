

import matplotlib.pyplot as plt
import os

def draw_or_save_loss_fig(train_losses, val_losses, save_path=None, is_show=False):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    if is_show:
        plt.show()
    plt.close(fig)
    

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_saving_fig_path(root_path, dataset_type, model_type, deactivate_feature, lr, optimizer_type):
    path = root_path + dataset_type + '_' + model_type + '_lr_' + str(lr) + '_optimizer_' + optimizer_type + '.png'
    if deactivate_feature:

        path += '_witout_' + str(deactivate_feature) + '.png'
    return path
