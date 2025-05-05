import json

def evaluate_nlvr(file_path):
    """
    Evaluate NLVR² predictions using simple accuracy.
    Expects a list of dicts with 'prediction' and 'label' fields.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    correct = sum(1 for item in data if item['prediction'] == item['label'])
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    dev_path = "nlvr_dev.json"
    testp_path = "nlvr_testP.json"

    dev_acc = evaluate_nlvr(dev_path)
    testp_acc = evaluate_nlvr(testp_path)

    print("======== NLVR² Evaluation ========")
    print(f"Dev Accuracy   : {dev_acc * 100:.2f}%")
    print(f"Test-P Accuracy: {testp_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
