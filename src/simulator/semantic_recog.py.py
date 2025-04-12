import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path
import random
import numpy as np
import datetime
import base64

# Add project root to path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Import utility functions
from block_agent.utils.utils import (
    save_to_json,
    load_from_json,
    save_base64_image,
    slugify
)

class SemanticRecognizability:
    """
    A class to evaluate how well generated 3D structures visually resemble their intended designs.
    Uses GPT-4 Vision as a vision-language evaluator to rank possible labels for each design.
    """
    
    def __init__(self, save_dir: str = "metrics/semantic_recognizability"):
        """
        Initialize the semantic recognizability evaluator.
        
        Args:
            save_dir (str): Directory to save evaluation results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize the LLM
        self.llm = ChatOpenAI(model="gpt-4o", max_tokens=1000)
        
        # Dictionary of related objects for each category
        self.related_objects = {
            "giraffe": ["elephant", "horse", "deer", "camel", "zebra", "llama", "antelope", "gazelle", "ostrich"],
            "table": ["chair", "desk", "cabinet", "shelf", "stand", "bench", "stool", "dresser", "counter"],
            "letter_u": ["letter_n", "letter_v", "letter_c", "letter_o", "letter_w", "letter_m", "letter_h", "letter_y", "letter_j"],
            "tree": ["bush", "plant", "flower", "cactus", "palm", "pine", "oak", "maple", "bamboo"],
            "bush": ["tree", "plant", "flower", "cactus", "palm", "pine", "oak", "maple", "bamboo"],
            "plant": ["tree", "bush", "flower", "cactus", "palm", "pine", "oak", "maple", "bamboo"],
            "flower": ["tree", "bush", "plant", "cactus", "palm", "pine", "oak", "maple", "bamboo"],
            "cactus": ["tree", "bush", "plant", "flower", "palm", "pine", "oak", "maple", "bamboo"],
            "palm": ["tree", "bush", "plant", "flower", "cactus", "pine", "oak", "maple", "bamboo"],
            "pine": ["tree", "bush", "plant", "flower", "cactus", "palm", "oak", "maple", "bamboo"],
            "oak": ["tree", "bush", "plant", "flower", "cactus", "palm", "pine", "maple", "bamboo"],
            "maple": ["tree", "bush", "plant", "flower", "cactus", "palm", "pine", "oak", "bamboo"],
            "bamboo": ["tree", "bush", "plant", "flower", "cactus", "palm", "pine", "oak", "maple"],
            # Add more categories as needed
        }
        
        # Generic distractors for unknown categories
        self.generic_distractors = [
            "table", "chair", "animal", "building", "letter", "sculpture",
            "bridge", "tower", "vehicle", "tree", "robot", "abstract_shape",
            "bush", "plant", "flower", "cactus", "palm", "pine", "oak", "maple", "bamboo"
        ]

    def generate_label_set(self, correct_label: str, n: int = 5) -> List[str]:
        """
        Generate a set of N labels including the correct one and N-1 distractors.
        
        Args:
            correct_label (str): The correct label for the design
            n (int): Total number of labels to generate
            
        Returns:
            List[str]: List of N labels with the correct one and N-1 distractors
        """
        base_label = slugify(correct_label)
        distractors = self.related_objects.get(base_label, self.generic_distractors)
        distractors = [d for d in distractors if d != base_label]
        
        selected_distractors = random.sample(distractors, min(n-1, len(distractors)))
        all_labels = [correct_label] + selected_distractors
        random.shuffle(all_labels)
        
        return all_labels

    def evaluate_single(self, image_path: str, correct_label: str, n_labels: int = 5) -> Dict:
        """
        Evaluate a single design's semantic recognizability.
        
        Args:
            image_path (str): Path to the rendered image
            correct_label (str): The intended design label
            n_labels (int): Number of labels to include in evaluation
            
        Returns:
            Dict: Evaluation metrics including top-1 accuracy, average rank, and relative rank
        """
        # Generate label set
        labels = self.generate_label_set(correct_label, n_labels)
        
        # Read the image file and convert to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Construct the messages
        messages = [
            SystemMessage(content="""You are an expert in analyzing 3D block-based structures. 
Your task is to rank a list of possible labels based on how well they match the given image.
Provide a numbered ranking from most likely to least likely, with brief explanations."""),
            
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": f"Please rank these possible labels from most likely to least likely for this block structure:\n{', '.join(labels)}\n\nProvide your ranking with brief explanations for each."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ])
        ]
        
        # Get the response from the LLM
        response = self.llm.invoke(messages)
        
        # Parse the ranking from the response
        correct_label_lower = correct_label.lower()
        ranking_text = response.content.lower()
        
        # Find the rank of the correct label
        rank = 1
        for line in ranking_text.split('\n'):
            if correct_label_lower in line:
                break
            if any(str(i) + '.' in line for i in range(1, n_labels + 1)):
                rank += 1
        
        # Calculate metrics
        top_1_accuracy = 1 if rank == 1 else 0
        relative_rank = (n_labels - rank + 1) / n_labels * 100
        
        result = {
            "n_labels": n_labels,
            "top_1_accuracy": top_1_accuracy,
            "average_rank": rank,
            "relative_rank": relative_rank,
            "full_response": response.content,
            "labels_used": labels,
            "image_path": image_path,
            "correct_label": correct_label,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save individual result
        result_filename = os.path.join(
            self.save_dir,
            f"single_{slugify(correct_label)}_{result['timestamp']}.json"
        )
        save_to_json(result, result_filename)
        
        return result

    def evaluate_batch(self, directory: str, label_mapping: Dict[str, str], n_labels: int = 5) -> Dict:
        """
        Evaluate multiple designs from a directory.
        
        Args:
            directory (str): Directory containing rendered images
            label_mapping (Dict[str, str]): Mapping of filenames to their correct labels
            n_labels (int): Number of labels to use in evaluation
            
        Returns:
            Dict: Aggregated evaluation metrics
        """
        results = []
        
        for file in Path(directory).glob("*.jpg"):
            if str(file) in label_mapping:
                try:
                    result = self.evaluate_single(
                        str(file),
                        label_mapping[str(file)],
                        n_labels
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        
        if not results:
            return {"error": "No results to aggregate"}
        
        # Aggregate metrics
        aggregated = {
            "n_labels": n_labels,
            "top_1_accuracy": np.mean([r["top_1_accuracy"] for r in results]) * 100,
            "average_rank": np.mean([r["average_rank"] for r in results]),
            "relative_rank": np.mean([r["relative_rank"] for r in results]),
            "individual_results": results,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_samples": len(results)
        }
        
        # Save aggregated results
        save_to_json(
            aggregated,
            os.path.join(self.save_dir, f"batch_results_{aggregated['timestamp']}.json")
        )
        
        return aggregated

def main():
    """Example usage of the semantic recognizability evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate semantic recognizability of block-based designs')
    parser.add_argument('--path', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--label', type=str, help='Correct label for single image evaluation')
    parser.add_argument('--n_labels', type=int, default=5, help='Number of labels to use in evaluation')
    parser.add_argument('--batch', action='store_true', help='Process entire directory')
    parser.add_argument('--save_dir', type=str, default='metrics/semantic_recognizability',
                      help='Directory to save evaluation results')
    args = parser.parse_args()
    
    try:
        evaluator = SemanticRecognizability(save_dir=args.save_dir)
        
        if args.batch:
            # Try to load label mappings from a JSON file
            try:
                label_mapping = load_from_json("label_mappings.json")
            except:
                # If no mapping file exists, use a default mapping
                label_mapping = {
                    "giraffe_001.jpg": "giraffe",
                    "table_001.jpg": "table",
                }
                save_to_json(label_mapping, "label_mappings.json")
            
            results = evaluator.evaluate_batch(args.path, label_mapping, args.n_labels)
            print("\nAggregated Results:")
            print(f"Number of labels: {results['n_labels']}")
            print(f"Total samples evaluated: {results['total_samples']}")
            print(f"Top-1 Accuracy: {results['top_1_accuracy']:.1f}%")
            print(f"Average Rank: {results['average_rank']:.2f}")
            print(f"Relative Rank: {results['relative_rank']:.1f}%")
        else:
            if not args.label:
                raise ValueError("Label must be provided for single image evaluation")
            
            result = evaluator.evaluate_single(args.path, args.label, args.n_labels)
            print("\nEvaluation Results:")
            print(f"Number of labels: {result['n_labels']}")
            print(f"Top-1 Accuracy: {100 if result['top_1_accuracy'] else 0}%")
            print(f"Average Rank: {result['average_rank']}")
            print(f"Relative Rank: {result['relative_rank']:.1f}%")
            print("\nFull Analysis:")
            print(result['full_response'])
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 