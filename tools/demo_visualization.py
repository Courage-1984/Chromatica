#!/usr/bin/env python3
"""
Demo script for the Chromatica visualization features.

This script demonstrates the new visual capabilities:
- Query color visualizations with weighted color bars
- Color palette representations
- Results collages with distance annotations
- Interactive color exploration

Usage:
    python tools/demo_visualization.py

The script provides practical examples of how to use the visualization
features for color-based image search applications.
"""

import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.visualization import (
    QueryVisualizer,
    ResultCollageBuilder,
    create_query_visualization,
    create_results_collage,
)


def demo_query_visualizations():
    """Demonstrate query visualization capabilities."""
    print("=" * 60)
    print("QUERY VISUALIZATION DEMO")
    print("=" * 60)

    # Test different color combinations and weights
    test_queries = [
        {
            "name": "Warm Colors",
            "colors": ["#FF0000", "#FFA500", "#FFFF00"],
            "weights": [0.5, 0.3, 0.2],
        },
        {
            "name": "Cool Colors",
            "colors": ["#0000FF", "#00FFFF", "#800080"],
            "weights": [0.4, 0.4, 0.2],
        },
        {
            "name": "Earth Tones",
            "colors": ["#8B4513", "#A0522D", "#CD853F", "#D2B48C"],
            "weights": [0.3, 0.3, 0.2, 0.2],
        },
        {
            "name": "High Contrast",
            "colors": ["#000000", "#FFFFFF", "#FF0000"],
            "weights": [0.4, 0.4, 0.2],
        },
        {
            "name": "Pastel Palette",
            "colors": ["#FFB6C1", "#98FB98", "#87CEEB", "#DDA0DD"],
            "weights": [0.25, 0.25, 0.25, 0.25],
        },
    ]

    visualizer = QueryVisualizer()

    for query in test_queries:
        print(f"\n--- {query['name']} ---")
        print(f"Colors: {query['colors']}")
        print(f"Weights: {query['weights']}")

        try:
            # Create weighted color bar
            start_time = time.time()
            color_bar = visualizer.create_weighted_color_bar(
                query["colors"], query["weights"]
            )
            end_time = time.time()

            print(f"Color bar shape: {color_bar.shape}")
            print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")

            # Create color palette
            start_time = time.time()
            palette = visualizer.create_color_palette(query["colors"], query["weights"])
            end_time = time.time()

            print(f"Palette shape: {palette.shape}")
            print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")

            # Create comprehensive visualization
            start_time = time.time()
            summary_img = visualizer.create_query_summary_image(
                query["colors"], query["weights"]
            )
            end_time = time.time()

            print(f"Summary image shape: {summary_img.shape}")
            print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")

            # Save the comprehensive visualization
            output_path = f"demo_{query['name'].lower().replace(' ', '_')}_viz.png"
            visualizer.save_image(summary_img, output_path)
            print(f"Saved visualization to: {output_path}")

        except Exception as e:
            print(f"ERROR: {str(e)}")

    print()


def demo_collage_building():
    """Demonstrate results collage capabilities."""
    print("=" * 60)
    print("RESULTS COLLAGE DEMO")
    print("=" * 60)

    # Create sample image paths and distances for demonstration
    # In a real scenario, these would come from actual search results
    sample_image_paths = [
        "datasets/test-dataset-20/7348262.jpg",
        "datasets/test-dataset-20/7348831.png",
        "datasets/test-dataset-20/7349035.jpg",
        "datasets/test-dataset-20/7349252.jpg",
        "datasets/test-dataset-20/7349286.jpg",
        "datasets/test-dataset-20/7349288.jpg",
        "datasets/test-dataset-20/7349498.jpg",
        "datasets/test-dataset-20/7349500.jpg",
    ]

    # Sample distances (lower = more similar)
    sample_distances = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"Sample images: {len(sample_image_paths)}")
    print(f"Sample distances: {sample_distances}")

    try:
        # Test with different collage configurations
        test_configs = [
            {"max_images_per_row": 3, "output_size": (900, 600)},
            {"max_images_per_row": 4, "output_size": (1200, 600)},
            {"max_images_per_row": 5, "output_size": (1500, 600)},
        ]

        for config in test_configs:
            print(
                f"\n--- Collage Config: {config['max_images_per_row']} images per row ---"
            )

            builder = ResultCollageBuilder(
                max_images_per_row=config["max_images_per_row"]
            )

            # Create basic collage
            start_time = time.time()
            collage = builder.create_results_collage(
                sample_image_paths, sample_distances, config["output_size"]
            )
            end_time = time.time()

            print(f"Basic collage shape: {collage.shape}")
            print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")

            # Create annotated collage
            start_time = time.time()
            annotated_collage = builder.create_distance_annotated_collage(
                sample_image_paths, sample_distances, config["output_size"]
            )
            end_time = time.time()

            print(f"Annotated collage shape: {annotated_collage.shape}")
            print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")

            # Save the annotated collage
            output_path = f"demo_collage_{config['max_images_per_row']}perrow.png"
            from PIL import Image

            img = Image.fromarray(annotated_collage)
            img.save(output_path)
            print(f"Saved collage to: {output_path}")

    except Exception as e:
        print(f"ERROR: {str(e)}")

    print()


def demo_utility_functions():
    """Demonstrate utility functions for visualization."""
    print("=" * 60)
    print("UTILITY FUNCTIONS DEMO")
    print("=" * 60)

    # Test colors and weights
    colors = ["#FF0000", "#00FF00", "#0000FF"]
    weights = [0.5, 0.3, 0.2]

    print(f"Test colors: {colors}")
    print(f"Test weights: {weights}")

    try:
        # Test create_query_visualization utility
        print("\n--- Testing create_query_visualization ---")
        start_time = time.time()
        viz_path = create_query_visualization(colors, weights, "demo_utility_viz.png")
        end_time = time.time()

        print(f"Visualization created: {viz_path}")
        print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")

        # Test create_results_collage utility
        print("\n--- Testing create_results_collage ---")
        # Use sample data
        sample_paths = ["datasets/test-dataset-20/7348262.jpg"] * 4
        sample_distances = [0.0, 0.1, 0.2, 0.3]

        start_time = time.time()
        collage_path = create_results_collage(
            sample_paths, sample_distances, "demo_utility_collage.png"
        )
        end_time = time.time()

        print(f"Collage created: {collage_path}")
        print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")

    except Exception as e:
        print(f"ERROR: {str(e)}")

    print()


def demo_performance_benchmark():
    """Demonstrate performance characteristics."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK DEMO")
    print("=" * 60)

    # Test with increasing numbers of colors
    color_counts = [1, 3, 5, 10]

    # Generate test colors (cycling through a base set)
    base_colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#008000",
        "#FFC0CB",
        "#A52A2A",
        "#808080",
    ]

    print(f"{'Colors':<8} {'Viz Time (ms)':<15} {'Collage Time (ms)':<18}")
    print("-" * 60)

    for count in color_counts:
        # Generate colors and weights
        colors = (base_colors * (count // len(base_colors) + 1))[:count]
        weights = [1.0 / count] * count

        # Measure visualization performance
        viz_times = []
        collage_times = []

        for _ in range(5):  # Run 5 times for reliable timing
            try:
                # Test visualization
                start_time = time.time()
                create_query_visualization(colors, weights, f"temp_viz_{count}.png")
                end_time = time.time()
                viz_times.append((end_time - start_time) * 1000)

                # Test collage (with sample data)
                sample_paths = ["datasets/test-dataset-20/7348262.jpg"] * min(count, 4)
                sample_distances = [0.1 * i for i in range(len(sample_paths))]

                start_time = time.time()
                create_results_collage(
                    sample_paths, sample_distances, f"temp_collage_{count}.png"
                )
                end_time = time.time()
                collage_times.append((end_time - start_time) * 1000)

            except Exception as e:
                print(f"ERROR with {count} colors: {str(e)}")
                break

        if viz_times and collage_times:
            avg_viz_time = np.mean(viz_times)
            avg_collage_time = np.mean(collage_times)

            print(f"{count:<8} {avg_viz_time:<15.2f} {avg_collage_time:<18.2f}")

        # Clean up temporary files
        for i in range(count):
            Path(f"temp_viz_{count}.png").unlink(missing_ok=True)
            Path(f"temp_collage_{count}.png").unlink(missing_ok=True)

    print()


def main():
    """Run all visualization demos."""
    try:
        print("🎨 CHROMATICA VISUALIZATION DEMO 🎨")
        print("=" * 60)
        print("This demo showcases the new visual features:")
        print("• Query color visualizations with weighted color bars")
        print("• Color palette representations")
        print("• Results collages with distance annotations")
        print("• Performance benchmarking")
        print("=" * 60)
    except UnicodeEncodeError:
        print("CHROMATICA VISUALIZATION DEMO")
        print("=" * 60)
        print("This demo showcases the new visual features:")
        print("• Query color visualizations with weighted color bars")
        print("• Color palette representations")
        print("• Results collages with distance annotations")
        print("• Performance benchmarking")
        print("=" * 60)

    try:
        # Run all demos
        demo_query_visualizations()
        demo_collage_building()
        demo_utility_functions()
        demo_performance_benchmark()

        try:
            print("✅ All visualization demos completed successfully!")
            print("\n📁 Generated files:")
            print("• demo_*.png - Query visualizations")
            print("• demo_collage_*.png - Results collages")
            print("\n🚀 Try the new API endpoints:")
            print("• GET /visualize/query?colors=FF0000,00FF00&weights=0.7,0.3")
            print("• GET /visualize/results?colors=FF0000,00FF00&weights=0.7,0.3&k=10")
        except UnicodeEncodeError:
            print("All visualization demos completed successfully!")
            print("\nGenerated files:")
            print("• demo_*.png - Query visualizations")
            print("• demo_collage_*.png - Results collages")
            print("\nTry the new API endpoints:")
            print("• GET /visualize/query?colors=FF0000,00FF00&weights=0.7,0.3")
            print("• GET /visualize/results?colors=FF0000,00FF00&weights=0.7,0.3&k=10")

    except Exception as e:
        try:
            print(f"\n❌ Demo failed: {str(e)}")
        except UnicodeEncodeError:
            print(f"\nDemo failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
