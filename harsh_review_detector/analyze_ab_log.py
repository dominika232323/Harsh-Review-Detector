from harsh_review_detector.config import SERVICE_LOGS
from json import loads

def get_log(path = SERVICE_LOGS):
    with open(path, "r") as f:
        log = f.readlines()
    for line in log:
        line = loads(line)
    return log

def analyze_log(log):
    base_true_positives = 0
    base_true_negatives = 0
    base_false_positives = 0
    base_false_negatives = 0
    advanced_true_positives = 0
    advanced_true_negatives = 0
    advanced_false_positives = 0
    advanced_false_negatives = 0
    for line in log:
        line_data = loads(line)
        if (isinstance(line_data, str)):
            continue
        if line_data["model_used"] == "base-model": 
            if line_data["prediction"] == 0:
                if line_data["true_label"] == 0:
                    base_true_negatives += 1
                else:
                    base_false_negatives += 1
            else:
                if line_data["true_label"] == 0:
                    base_false_positives += 1
                else:
                    base_true_positives += 1

        else:
            if line_data["prediction"] == 0:
                if line_data["true_label"] == 0:
                    advanced_true_negatives += 1
                else:
                    advanced_false_negatives += 1
            else:
                if line_data["true_label"] == 0:
                    advanced_false_positives += 1
                else:
                    advanced_true_positives += 1
    print(f"Base model handled {base_false_negatives+base_false_positives+base_true_negatives+base_true_positives} cases.")
    print(f"There were {base_true_positives + base_true_negatives} correct predictions, including {base_true_positives} true positives and {base_true_negatives} true negatives.")
    print(f"There were {base_false_positives + base_false_negatives} incorrect predictions, including {base_false_positives} false positives and {base_false_negatives} false negatives.\n")
    base_accuracy = (base_true_positives + base_true_negatives)/(base_true_positives + base_true_negatives+base_false_positives + base_false_negatives)
    base_precision = base_true_positives/(base_true_positives + base_false_positives)
    base_recall = base_true_positives/(base_true_positives+base_false_negatives)
    base_f1 = 2 * (base_precision * base_recall) / (base_precision + base_recall)
    print(f"Accuracy of the base model is {base_accuracy}")
    print(f"Precision of the base model is {base_precision}")
    print(f"Recall of the base model is {base_recall}")
    print(f"F1 score of the base model is {base_f1}")

    print(f"Advanced model handled {advanced_false_negatives+advanced_false_positives+advanced_true_negatives+advanced_true_positives} cases.")
    print(f"There were {advanced_true_positives + advanced_true_negatives} correct predictions, including {advanced_true_positives} true positives and {advanced_true_negatives} true negatives.")
    print(f"There were {advanced_false_positives + advanced_false_negatives} incorrect predictions, including {advanced_false_positives} false positives and {advanced_false_negatives} false negatives.")
    
    advanced_accuracy = (advanced_true_positives + advanced_true_negatives)/(advanced_true_positives + advanced_true_negatives+advanced_false_positives + advanced_false_negatives)
    advanced_precision = advanced_true_positives/(advanced_true_positives + advanced_false_positives)
    advanced_recall = advanced_true_positives/(advanced_true_positives+advanced_false_negatives)
    advanced_f1 = 2 * (advanced_precision * advanced_recall) / (advanced_precision + advanced_recall)
    print(f"Accuracy of the advanced model is {advanced_accuracy}")
    print(f"Precision of the advanced model is {advanced_precision}")
    print(f"Recall of the advanced model is {advanced_recall}")
    print(f"F1 score of the advanced model is {advanced_f1}")


if __name__ == "__main__":
    analyze_log(get_log())