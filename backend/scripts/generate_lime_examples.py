"""
Generate 10 LIME Example Visualizations

This script generates LIME explanations and visualizations for 10 example
financial sentiment texts. This fulfills the FYP-159 requirement for
"Visualisations for 10 examples".

Usage:
    python backend/scripts/generate_lime_examples.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.explainability.lime_explainer import get_lime_explainer
from app.explainability.visualizations import (
    plot_lime_features,
    save_lime_html,
)

# 10 diverse financial sentiment examples
EXAMPLE_TEXTS = [
    # Positive sentiment examples
    "Stock prices surged to record highs as investors celebrated strong earnings",
    "The company reported exceptional quarterly results with revenue growth exceeding expectations",
    "Markets rallied on positive economic data showing robust job growth",
    "Investor confidence soared following the announcement of a major business expansion",
    
    # Negative sentiment examples
    "Share prices plummeted amid fears of an impending recession",
    "The company's quarterly losses exceeded analyst predictions, causing widespread concern",
    "Markets crashed following disappointing employment figures and weak economic indicators",
    "Stock values tumbled as investors reacted to news of bankruptcy proceedings",
    
    # Neutral/Mixed sentiment examples
    "The Federal Reserve announced its decision to maintain current interest rates",
    "Quarterly earnings report released showing mixed results across different sectors",
]


def generate_examples():
    """Generate LIME explanations and visualizations for 10 examples."""
    print("=" * 70)
    print("LIME EXPLAINABILITY - GENERATING 10 EXAMPLES")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("data/processed/explanations/lime_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print("\nInitializing LIME explainer...")
    
    # Initialize LIME explainer
    explainer = get_lime_explainer(num_features=10, num_samples=1000)
    
    print("[OK] LIME explainer initialized\n")
    
    # Generate explanations for each example
    print("Generating explanations and visualizations...")
    print("-" * 70)
    
    explanations = []
    
    for i, text in enumerate(EXAMPLE_TEXTS, 1):
        print(f"\n[{i}/10] Processing text:")
        print(f"   Text: {text[:60]}..." if len(text) > 60 else f"   Text: {text}")
        
        try:
            # Generate explanation
            explanation = explainer.explain(text, num_features=10)
            explanations.append(explanation)
            
            predicted_label = explanation["prediction"]["label"]
            confidence = explanation["prediction"]["score"]
            print(f"   Prediction: {predicted_label.upper()} ({confidence:.1%})")
            
            # Save visualization as PNG
            png_filename = f"example_{i:02d}_{predicted_label}_{timestamp}.png"
            png_path = output_dir / png_filename
            plot_lime_features(
                explanation,
                save_path=str(png_path),
                show=False,
                max_features=10,
            )
            print(f"   [OK] Saved PNG: {png_filename}")
            
            # Save HTML explanation
            html_filename = f"example_{i:02d}_{predicted_label}_{timestamp}.html"
            html_path = output_dir / html_filename
            save_lime_html(explanation, html_path)
            print(f"   [OK] Saved HTML: {html_filename}")
            
            # Display top features
            top_features = explanation["top_features"][:3]
            print(f"   Top features: {', '.join([f[0] for f in top_features])}")
            
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            explanations.append(None)
    
    print("\n" + "-" * 70)
    
    # Summary statistics
    successful = sum(1 for e in explanations if e is not None)
    print(f"\n[OK] Generated {successful}/10 examples successfully")
    
    # Prediction distribution
    if successful > 0:
        valid_explanations = [e for e in explanations if e is not None]
        positive_count = sum(
            1 for e in valid_explanations if e["prediction"]["label"] == "positive"
        )
        negative_count = sum(
            1 for e in valid_explanations if e["prediction"]["label"] == "negative"
        )
        neutral_count = sum(
            1 for e in valid_explanations if e["prediction"]["label"] == "neutral"
        )
        
        print("\nPrediction Distribution:")
        print(f"  Positive: {positive_count}")
        print(f"  Negative: {negative_count}")
        print(f"  Neutral:  {neutral_count}")
    
    print(f"\n[DONE] All visualizations saved to: {output_dir}")
    print("=" * 70)
    
    return explanations


def main():
    """Main entry point."""
    try:
        explanations = generate_examples()
        
        # Return success if at least 8/10 succeeded
        successful = sum(1 for e in explanations if e is not None)
        if successful >= 8:
            print("\n[SUCCESS] LIME examples generated successfully")
            return 0
        else:
            print(f"\n[WARNING] Only {successful}/10 examples succeeded")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
