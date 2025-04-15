import torch
import torch.nn.functional as F
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup


def eval_all(pred, target, idx_split):
    result = {}
    with torch.no_grad():
        y_pred = pred
        y_true = target
        for split_type in ["train", "valid", "test"]:
            y_pred_s = y_pred[idx_split[split_type]]
            y_true_s = y_true[idx_split[split_type]]
            result[split_type] = (y_pred_s == y_true_s).sum().cpu() / y_true_s.shape[0]
    return result

def predict(model, graph):
    model.eval()
    with torch.no_grad():
        y_pred =  model(graph).argmax(dim=1)
    model.train()
    return y_pred


def train(model, graph, idx_split, config):
    # move to device
    graph = graph.to(config.device)
    graph.edge_index = graph.edge_index.to(config.device)
    graph.y = graph.y.to(config.device)
    idx_split["train"] = idx_split["train"].to(config.device)
    idx_split["valid"] = idx_split["valid"].to(config.device)
    model = model.to(config.device)

    model.train()
    best_model = None
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(optim, 20, config.epoches, min_lr=0.1 * config.lr)
    best_eval = 0.0
    best_step = 0
    for i in range(config.epoches):
        y_pred =  model(graph)
        loss = F.cross_entropy(y_pred[idx_split["train"]], graph.y[idx_split["train"]])

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        y_pred = predict(model, graph)
        result = eval_all(y_pred, graph.y, idx_split)
        print(f"\r[Epoch {i}]  loss: {loss.item():.4f}  eval: {result}", end="")
        eval_val = result["valid"]

        if i % config.print_freq == 0:
            print()
        if eval_val > best_eval:
            best_eval = eval_val
            best_model = {k:v.clone() for k,v in model.state_dict().items()}
            #print(f"update, best_model = {best_model['in_conv.lin.weight'][:10,0]}")
            best_step = i

    if best_model is not None:
        print(f"\nCurrent model {result}")
        model.load_state_dict(best_model)
        print(f"\nBest model on {best_step} step {best_eval}, {eval_all(predict(model, graph), graph.y, idx_split)}")
