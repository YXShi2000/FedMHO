import pcode.aggregation.utils as agg_utils
import torch
from copy import deepcopy


def aggregate(
        conf,
        fedavg_models,
        client_models,
        criterion,
        metrics,
        flatten_local_models,
        test_data_loader,
):
    # recover the models on the computation device.
    _, local_models = agg_utils.recover_models(
        conf, client_models, flatten_local_models
    )

    device = torch.device("cuda") if conf.graph.on_cuda else torch.device("cpu")

    # fedavg_models_lst = [fedavg_model.to(device) for _, fedavg_model in fedavg_models.items()]
    correct, total = 0, 0
    for batch_idx, img_target in enumerate(test_data_loader):
        img, target = img_target
        img, target = img.to(device), target.to(device)

        # ensemble_logits = torch.zeros_like(fedavg_models_lst[0](img), requires_grad=False).to(device)
        # for fedavg_model in fedavg_models_lst:
        #     fedavg_model.eval()
        #     with torch.no_grad():
        #         ensemble_logits += fedavg_model(img)

        # Optimize GPU usage
        cnt = 0
        ensemble_logits = 0
        for _, fedavg_model in fedavg_models.items():
            fedavg_model.to(device)
            fedavg_model.eval()
            with torch.no_grad():
                if cnt == 0:
                    ensemble_logits = fedavg_model(img)
                else:
                    ensemble_logits += fedavg_model(img)
            del fedavg_model
            cnt += 1

        ensemble_logits /= cnt

        _, predicted = torch.max(ensemble_logits, dim=1)
        total += target.size(0)
        correct += torch.eq(predicted, target).sum().item()

    epoch_acc = 100 * correct / total
    print("accuracy: %.3f %% " % epoch_acc)

    _client_models = {}
    for arch, model in fedavg_models.items():
        _client_models[arch] = deepcopy(model).cpu()

    return _client_models, epoch_acc
