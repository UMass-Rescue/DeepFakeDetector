import pandas as pd
import os
from typing import List, Optional
from detect_deepfakes import run_detection
from tqdm import tqdm

def find_video_files(search_path: str, extensions: Optional[List[str]] = None) -> List[str]:
    """Recursively search for video files in the given directory."""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg']

    video_files = []
    for root, _, files in os.walk(search_path):
        video_files.extend(
            os.path.abspath(os.path.join(root, f))
            for f in files if any(f.lower().endswith(ext) for ext in extensions)
        )
    return video_files

def process_video_directory(search_path: str, model: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    """Process videos in the specified directory and save results."""
    video_paths = find_video_files(search_path)
    print(f"Found {len(video_paths)} video files in {search_path}")

    results_df = run_detection(video_paths, model=model, real_threshold=0.2, fake_threshold=0.8)
    
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    return results_df

def count_predictions(results_df: pd.DataFrame) -> pd.Series:
    """Count the number of each prediction type."""
    return results_df['prediction'].value_counts()

def main():
    """Main entry point of the script."""
    search_directory = input("Enter the path to the video directory: ").strip()
    models = ['EfficientNetB4', 'EfficientNetB4ST', 'EfficientNetAutoAttB4', 'EfficientNetAutoAttB4ST']

    for model in tqdm(models):
        output_csv = f'deepfake_results_{model}.csv'
        results_df = process_video_directory(search_directory, model=model, output_csv=output_csv)
        prediction_counts = count_predictions(results_df)

        # Display the counts
        print(f"\nPrediction Counts for {model}:")
        print(prediction_counts)
        print("-" * 40)

if __name__ == "__main__":
    main()
