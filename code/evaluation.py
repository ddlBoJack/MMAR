import argparse
import json
import pickle
from tqdm import tqdm
from pathlib import Path
import re

def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process benchmark JSON and calculate accuracy.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file to be evaluated')
    
    args = parser.parse_args()  
    
    if args.input.endswith('json'):
        with open(args.input, 'r') as f:
            input_data = json.load(f)
    elif args.input.endswith('jsonl'):
        with open(args.input, 'r') as f:
            input_data = [json.loads(line.strip()) for line in f]

    corr, total = 0, 0

    # Track metrics for different categories:
    modality_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0], 'mix-sound-music': [0, 0], 'mix-sound-speech': [0, 0], 'mix-music-speech': [0, 0], 'mix-sound-music-speech': [0, 0]}
    category_metrics = {'Signal Layer': [0, 0], 'Perception Layer': [0, 0], 'Semantic Layer': [0, 0], 'Cultural Layer': [0, 0]}
    
    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    output_key = 'answer_prediction' # The key that contains model output
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    # for idx, sample in enumerate(tqdm(input_data)):
    for idx, sample in enumerate(input_data):
        
        # If there's no model output key, skip
        if output_key not in sample:
            continue
        
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        modality = sample['modality']
        category = sample['category']
        choices = sample['choices']
        
        # Get the sub-category
        subcat = sample.get('sub-category', None)
        if subcat is not None:
            # If we haven't seen this sub-category before, initialize
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            modality_metrics[modality][0] += 1
            category_metrics[category][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        new_data.append(sample)
        modality_metrics[modality][1] += 1
        category_metrics[category][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1


    # Print results:
    print("*"*30)
    print("Modality-wise Accuracy:")
    for modality in modality_metrics:
        n_correct, n_total = modality_metrics[modality]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{modality} : {acc:.2f}% over {n_total} samples")
    
    print("*"*30)
    print("Category-wise Accuracy:")
    for category in category_metrics:
        n_correct, n_total = category_metrics[category]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{category} : {acc:.2f}% over {n_total} samples")
    
    print("*"*30)
    print("Sub-category-wise Accuracy:")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*"*30)
    print(f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples")
    print("*"*30)
    print(f"No prediction count: {no_pred_count}")
