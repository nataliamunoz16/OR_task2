import os
import json
import glob


def extract_epoch_mdice(item):
    epoch = item.get("epoch")
    mdice = item.get("mDice")
    return epoch, mdice


def select_best_epoch(list_epochs):
    best = None
    best_mdice = float("-inf")
    best_epoch = float("-inf")
    for item in list_epochs:
        if not isinstance(item, dict):
            continue
        epoch, mdice=extract_epoch_mdice(item)
        if epoch is None or mdice is None:
            continue
        if mdice > best_mdice or (mdice == best_mdice and epoch>best_epoch):
            best=item
            best_mdice=mdice
            best_epoch=epoch
    return best


def proces_jsons(directory=".", prefix="model_name", sufix="_validation_metrics.json", output="best_results_lr_batchsize_resolution.json"):
    pattern= os.path.join(directory, f"{prefix}*{sufix}")
    files= glob.glob(pattern)
    results= {}
    for path in files:
        name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data =json.load(f)
            if not isinstance(data, list):
                results[name] = {"error": "JSON has no list format"}
                continue
            best = select_best_epoch(data)
            if best is None:
                results[name] = {"error": "No valid epoch with mDice"}
            else:
                results[name] = best

        except Exception as e:
            results[name] = {"error": str(e)}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"results saved in: {output}")


if __name__ == "__main__":
    proces_jsons(
        directory="results",
        prefix="deeplabv3+",
        sufix="_validation_metrics.json",
        output="deeplabv3+_best_results.json"
    )