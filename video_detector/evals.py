import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def load_and_process_results(file_paths):
    """
    Load and process results from multiple JSON files.
    
    Args:
        file_paths (list): List of file paths to process
    
    Returns:
        pd.DataFrame: Processed results DataFrame
    """
    all_results = []
    
    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract model name and true label from the filename
            file_name = os.path.basename(file_path)
            parts = file_name.split("_")
            true_label = "real" if "real" in parts[1] else "fake"
            model_name = parts[2].replace("results.json", "").strip()
            
            # Parse JSON and append to results list
            for entry in data:
                result_entry = {
                    "input_path": entry.get("input_path", ""),
                    "frames_analyzed": entry.get("frames_analyzed", 0),
                    "frames_with_faces": entry.get("frames_with_faces", 0),
                    "final_label": entry.get("final_label", ""),
                    "confidence_real": entry.get("confidence_scores", {}).get("real", 0),
                    "confidence_fake": entry.get("confidence_scores", {}).get("fake", 0),
                    "average_prediction": entry.get("average_prediction", 0),
                    "true_label": true_label,
                    "model": model_name
                }
                all_results.append(result_entry)
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(all_results)

def analyze_model_performance(results_df):
    """
    Analyze model performance with comprehensive metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results
    
    Returns:
        dict: Dictionary containing performance metrics for each model
    """
    # Prepare performance metrics dictionary
    performance_metrics = {}
    
    # Get unique models
    models = results_df['model'].unique()
    
    # Performance metrics visualization
    plt.figure(figsize=(20, 15))
    
    for i, model in enumerate(models, 1):
        # Filter results for current model
        model_results = results_df[results_df['model'] == model]
        
        # Prepare labels
        true_labels = model_results['true_label']
        predicted_labels = model_results['final_label']
        
        # Generate classification report
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=['fake', 'real'])
        
        # Store metrics
        performance_metrics[model] = {
            'Detailed_Metrics': report,
            'Confusion_Matrix': {
                'Fake_Fake': cm[0, 0],
                'Fake_Real': cm[0, 1],
                'Real_Fake': cm[1, 0],
                'Real_Real': cm[1, 1]
            },
            'Total_Samples': len(model_results)
        }
        
        # Visualization Subplots
        plt.subplot(len(models), 2, 2*i-1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fake', 'Real'], 
                    yticklabels=['Fake', 'Real'])
        plt.title(f'{model} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Bar plot of precision, recall, f1-score
        plt.subplot(len(models), 2, 2*i)
        metrics_df = pd.DataFrame({
            'Precision': [report['fake']['precision'], report['real']['precision']],
            'Recall': [report['fake']['recall'], report['real']['recall']],
            'F1-Score': [report['fake']['f1-score'], report['real']['f1-score']]
        }, index=['Fake', 'Real'])
        metrics_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'{model} Performance Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed performance metrics
    print("\nDetailed Performance Metrics:")
    for model, metrics in performance_metrics.items():
        print(f"\nModel: {model}")
        print("\nClassification Report:")
        report = metrics['Detailed_Metrics']
        
        # Format and print classification report
        print(f"Fake Class:")
        print(f"  Precision: {report['fake']['precision']:.4f}")
        print(f"  Recall: {report['fake']['recall']:.4f}")
        print(f"  F1-Score: {report['fake']['f1-score']:.4f}")
        print(f"  Support: {report['fake']['support']}")
        
        print(f"\nReal Class:")
        print(f"  Precision: {report['real']['precision']:.4f}")
        print(f"  Recall: {report['real']['recall']:.4f}")
        print(f"  F1-Score: {report['real']['f1-score']:.4f}")
        print(f"  Support: {report['real']['support']}")
        
        print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = metrics['Confusion_Matrix']
        print(f"  Fake predicted as Fake: {cm['Fake_Fake']}")
        print(f"  Fake predicted as Real: {cm['Fake_Real']}")
        print(f"  Real predicted as Fake: {cm['Real_Fake']}")
        print(f"  Real predicted as Real: {cm['Real_Real']}")
    
    return performance_metrics

def main():
    # Define file paths (modify as needed)
    file_paths = [
        # "/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results/videos_fake18_results.json",
        # "/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results/videos_real18_results.json",
        "/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results/videos_fakeX_results.json",
        "/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results/videos_realX_results.json",
        # "/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results/videos_fakeres50_results.json",
        # "/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results/videos_realres50_results.json",
    ]
    
    # Load and process results
    all_results = load_and_process_results(file_paths)
    
    # Analyze performance
    model_performance = analyze_model_performance(all_results)
    
    # Optional: Save results to CSV
    all_results.to_csv("combined_results.csv", index=False)
    
    # Convert performance metrics to a more readable format for CSV export
    performance_data = []
    for model, metrics in model_performance.items():
        model_metrics = metrics['Detailed_Metrics']
        performance_entry = {
            'Model': model,
            'Fake_Precision': model_metrics['fake']['precision'],
            'Fake_Recall': model_metrics['fake']['recall'],
            'Fake_F1_Score': model_metrics['fake']['f1-score'],
            'Fake_Support': model_metrics['fake']['support'],
            'Real_Precision': model_metrics['real']['precision'],
            'Real_Recall': model_metrics['real']['recall'],
            'Real_F1_Score': model_metrics['real']['f1-score'],
            'Real_Support': model_metrics['real']['support'],
            'Overall_Accuracy': model_metrics['accuracy'],
            'Fake_Predicted_Fake': metrics['Confusion_Matrix']['Fake_Fake'],
            'Fake_Predicted_Real': metrics['Confusion_Matrix']['Fake_Real'],
            'Real_Predicted_Fake': metrics['Confusion_Matrix']['Real_Fake'],
            'Real_Predicted_Real': metrics['Confusion_Matrix']['Real_Real']
        }
        performance_data.append(performance_entry)
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv("model_performance_metrics.csv", index=False)

if __name__ == "__main__":
    main()