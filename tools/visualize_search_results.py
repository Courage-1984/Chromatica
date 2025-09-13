#!/usr/bin/env python3
"""
Search Results Visualization Tool for Chromatica.

This tool provides comprehensive visualization of search results, including:
- Result ranking analysis
- Distance distribution visualization
- Color similarity mapping
- Performance metrics analysis
- Interactive result exploration

Features:
- Visualize search result rankings and distances
- Analyze color similarity patterns
- Generate performance reports
- Create interactive result galleries
- Export visualizations for analysis

Usage:
    python tools/visualize_search_results.py --results search_results.json
    python tools/visualize_search_results.py --api-query "FF0000" --k 10
    python tools/visualize_search_results.py --compare query1.json query2.json
"""

import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import requests
import seaborn as sns
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up matplotlib for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class SearchResultsVisualizer:
    """Comprehensive search results visualization and analysis tool."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the search results visualizer.
        
        Args:
            api_base_url: Base URL for the Chromatica API
        """
        self.api_base_url = api_base_url
        self.colormap = plt.cm.viridis
        
    def query_api(self, colors: List[str], weights: List[float], k: int = 10) -> Dict[str, Any]:
        """
        Query the Chromatica API for search results.
        
        Args:
            colors: List of hex color codes
            weights: List of color weights
            k: Number of results to retrieve
            
        Returns:
            API response as dictionary
        """
        try:
            # Prepare query parameters
            colors_str = ",".join(colors)
            weights_str = ",".join(f"{w:.3f}" for w in weights)
            
            # Make API request
            url = f"{self.api_base_url}/search"
            params = {
                "colors": colors_str,
                "weights": weights_str,
                "k": k
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error querying API: {e}")
            return {}
    
    def load_results_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load search results from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Search results as dictionary
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results from {file_path}: {e}")
            return {}
    
    def create_ranking_visualization(self, results: List[Dict], title: str = "Search Results Ranking",
                                   save_path: Optional[str] = None) -> None:
        """
        Create a visualization of search result rankings.
        
        Args:
            results: List of search result dictionaries
            title: Title for the visualization
            save_path: Optional path to save the image
        """
        if not results:
            print("No results to visualize")
            return
        
        # Extract data
        image_ids = [result['image_id'] for result in results]
        distances = [result['distance'] for result in results]
        ranks = list(range(1, len(results) + 1))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Ranking bar chart
        bars = ax1.bar(ranks, distances, color=self.colormap(np.linspace(0, 1, len(ranks))))
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Distance (Lower = More Similar)')
        ax1.set_title('Search Results by Rank')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, distance in zip(bars, distances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(distances) * 0.01,
                    f'{distance:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Distance distribution
        ax2.hist(distances, bins=min(10, len(distances)), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distance Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        ax2.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.2f}')
        ax2.axvline(mean_dist + std_dist, color='orange', linestyle=':', label=f'+1σ: {mean_dist + std_dist:.2f}')
        ax2.axvline(mean_dist - std_dist, color='orange', linestyle=':', label=f'-1σ: {mean_dist - std_dist:.2f}')
        ax2.legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Ranking visualization saved to: {save_path}")
        
        plt.show()
    
    def create_color_similarity_heatmap(self, results: List[Dict], title: str = "Color Similarity Analysis",
                                      save_path: Optional[str] = None) -> None:
        """
        Create a heatmap showing color similarity patterns.
        
        Args:
            results: List of search result dictionaries
            title: Title for the visualization
            save_path: Optional path to save the image
        """
        if not results:
            print("No results to visualize")
            return
        
        # Extract dominant colors and create similarity matrix
        all_colors = []
        for result in results:
            colors = result.get('dominant_colors', [])
            all_colors.extend(colors)
        
        # Remove duplicates and limit to reasonable number
        unique_colors = list(set(all_colors))[:20]  # Limit to 20 colors for readability
        
        if len(unique_colors) < 2:
            print("Not enough unique colors for similarity analysis")
            return
        
        # Create similarity matrix (simple RGB distance for demo)
        similarity_matrix = np.zeros((len(unique_colors), len(unique_colors)))
        
        for i, color1 in enumerate(unique_colors):
            for j, color2 in enumerate(unique_colors):
                if i == j:
                    similarity_matrix[i, j] = 0
                else:
                    # Convert hex to RGB and calculate distance
                    rgb1 = self.hex_to_rgb(color1)
                    rgb2 = self.hex_to_rgb(color2)
                    distance = np.linalg.norm(np.array(rgb1) - np.array(rgb2))
                    similarity_matrix[i, j] = distance
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # Add color labels
        ax.set_xticks(range(len(unique_colors)))
        ax.set_yticks(range(len(unique_colors)))
        ax.set_xticklabels(unique_colors, rotation=45, ha='right')
        ax.set_yticklabels(unique_colors)
        
        # Add color patches
        for i, color in enumerate(unique_colors):
            rect = patches.Rectangle((i-0.5, -0.5), 1, 1, facecolor=color, 
                                   edgecolor='white', linewidth=2)
            ax.add_patch(rect)
        
        # Add value annotations
        for i in range(len(unique_colors)):
            for j in range(len(unique_colors)):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.0f}',
                             ha="center", va="center", color="white", fontweight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Colors')
        ax.set_ylabel('Colors')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RGB Distance (Lower = More Similar)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Similarity heatmap saved to: {save_path}")
        
        plt.show()
    
    def create_performance_analysis(self, results: List[Dict], metadata: Dict, 
                                  title: str = "Search Performance Analysis",
                                  save_path: Optional[str] = None) -> None:
        """
        Create a performance analysis visualization.
        
        Args:
            results: List of search result dictionaries
            metadata: Search metadata dictionary
            title: Title for the visualization
            save_path: Optional path to save the image
        """
        if not results:
            print("No results to analyze")
            return
        
        # Extract performance metrics
        total_time = metadata.get('total_time_ms', 0)
        ann_time = metadata.get('ann_time_ms', 0)
        rerank_time = metadata.get('rerank_time_ms', 0)
        results_count = metadata.get('results_count', len(results))
        
        # Create performance breakdown
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Timing breakdown pie chart
        timing_data = [ann_time, rerank_time, total_time - ann_time - rerank_time]
        timing_labels = ['ANN Search', 'Reranking', 'Other']
        timing_colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        wedges, texts, autotexts = ax1.pie(timing_data, labels=timing_labels, autopct='%1.1f%%',
                                           colors=timing_colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('Search Time Breakdown')
        
        # Performance metrics bar chart
        metrics = ['Total Time', 'ANN Time', 'Rerank Time']
        values = [total_time, ann_time, rerank_time]
        bars = ax2.bar(metrics, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Performance Metrics')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                    f'{value:.0f}ms', ha='center', va='bottom', fontsize=10)
        
        # Distance vs rank scatter plot
        ranks = list(range(1, len(results) + 1))
        distances = [result['distance'] for result in results]
        
        ax3.scatter(ranks, distances, c=distances, cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Distance')
        ax3.set_title('Distance vs Rank')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(ranks, distances, 1)
        p = np.poly1d(z)
        ax3.plot(ranks, p(ranks), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax3.legend()
        
        # Results summary table
        ax4.axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Total Results', str(results_count)],
            ['Total Time', f'{total_time:.0f} ms'],
            ['ANN Time', f'{ann_time:.0f} ms'],
            ['Rerank Time', f'{rerank_time:.0f} ms'],
            ['Avg Distance', f'{np.mean(distances):.2f}'],
            ['Min Distance', f'{min(distances):.2f}'],
            ['Max Distance', f'{max(distances):.2f}']
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style the header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4ecdc4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Search Summary', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance analysis saved to: {save_path}")
        
        plt.show()
    
    def create_result_gallery(self, results: List[Dict], query_colors: List[str] = None,
                           title: str = "Search Results Gallery",
                           save_path: Optional[str] = None) -> None:
        """
        Create a visual gallery of search results.
        
        Args:
            results: List of search result dictionaries
            query_colors: List of query colors for reference
            title: Title for the visualization
            save_path: Optional path to save the image
        """
        if not results:
            print("No results to display")
            return
        
        # Calculate grid dimensions
        n_results = len(results)
        cols = min(5, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Add query colors at the top if provided
        if query_colors:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            
            # Create query color bar
            query_ax = fig.add_axes([0.1, 0.92, 0.8, 0.05])
            for i, color in enumerate(query_colors):
                rect = patches.Rectangle((i * 0.2, 0), 0.18, 1, facecolor=color, 
                                       edgecolor='black', linewidth=2)
                query_ax.add_patch(rect)
                query_ax.text(i * 0.2 + 0.09, 0.5, color, ha='center', va='center', 
                            fontsize=10, fontfamily='monospace', color='white', fontweight='bold')
            query_ax.set_xlim(0, len(query_colors) * 0.2)
            query_ax.set_ylim(0, 1)
            query_ax.axis('off')
            query_ax.set_title('Query Colors', fontsize=12, fontweight='bold')
        else:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create result cards
        for i, result in enumerate(results):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Create result card
            self._create_result_card(ax, result, i + 1)
        
        # Hide empty subplots
        for i in range(n_results, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Result gallery saved to: {save_path}")
        
        plt.show()
    
    def _create_result_card(self, ax, result: Dict, rank: int) -> None:
        """Create a single result card for the gallery."""
        ax.axis('off')
        
        # Result header
        ax.text(0.5, 0.95, f"Rank {rank}", ha='center', va='top', 
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        # Image ID
        ax.text(0.5, 0.88, f"ID: {result['image_id']}", ha='center', va='top',
               fontsize=10, fontfamily='monospace', transform=ax.transAxes)
        
        # Distance
        distance = result.get('distance', 0)
        ax.text(0.5, 0.8, f"Distance: {distance:.2f}", ha='center', va='top',
               fontsize=10, transform=ax.transAxes)
        
        # Dominant colors
        colors = result.get('dominant_colors', [])
        if colors:
            ax.text(0.5, 0.7, "Dominant Colors:", ha='center', va='top',
                   fontsize=10, fontweight='bold', transform=ax.transAxes)
            
            # Create color swatches
            for i, color in enumerate(colors[:5]):  # Limit to 5 colors
                x_pos = 0.2 + i * 0.15
                if x_pos < 0.9:  # Ensure it fits
                    rect = patches.Rectangle((x_pos, 0.55), 0.12, 0.1, 
                                           facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x_pos + 0.06, 0.52, color, ha='center', va='top',
                           fontsize=8, fontfamily='monospace', transform=ax.transAxes)
        
        # File path (truncated)
        file_path = result.get('file_path', '')
        if file_path:
            filename = Path(file_path).name
            ax.text(0.5, 0.4, f"File: {filename}", ha='center', va='top',
                   fontsize=8, transform=ax.transAxes, style='italic')
        
        # Add border
        border = patches.Rectangle((0.02, 0.02), 0.96, 0.96, 
                                 fill=False, edgecolor='gray', linewidth=2)
        ax.add_patch(border)
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def create_comprehensive_report(self, results: List[Dict], metadata: Dict,
                                  query_colors: List[str] = None,
                                  output_dir: str = "search_reports") -> None:
        """
        Create a comprehensive search results report.
        
        Args:
            results: List of search result dictionaries
            metadata: Search metadata dictionary
            query_colors: List of query colors
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(24, 20))
        
        # Query colors (if provided)
        if query_colors:
            ax_query = plt.subplot(4, 3, 1)
            ax_query.axis('off')
            for i, color in enumerate(query_colors):
                rect = patches.Rectangle((i * 0.2, 0), 0.18, 1, facecolor=color, 
                                       edgecolor='black', linewidth=2)
                ax_query.add_patch(rect)
                ax_query.text(i * 0.2 + 0.09, 0.5, color, ha='center', va='center', 
                            fontsize=12, fontfamily='monospace', color='white', fontweight='bold')
            ax_query.set_xlim(0, len(query_colors) * 0.2)
            ax_query.set_ylim(0, 1)
            ax_query.set_title('Query Colors', fontsize=14, fontweight='bold')
        
        # Ranking visualization
        if results:
            ranks = list(range(1, len(results) + 1))
            distances = [result['distance'] for result in results]
            
            ax_rank = plt.subplot(4, 3, 2)
            bars = ax_rank.bar(ranks, distances, color=self.colormap(np.linspace(0, 1, len(ranks))))
            ax_rank.set_xlabel('Rank')
            ax_rank.set_ylabel('Distance')
            ax_rank.set_title('Results by Rank')
            ax_rank.grid(True, alpha=0.3)
            
            # Distance distribution
            ax_dist = plt.subplot(4, 3, 3)
            ax_dist.hist(distances, bins=min(10, len(distances)), alpha=0.7, color='skyblue', edgecolor='black')
            ax_dist.set_xlabel('Distance')
            ax_dist.set_ylabel('Frequency')
            ax_dist.set_title('Distance Distribution')
            ax_dist.grid(True, alpha=0.3)
            
            # Performance metrics
            total_time = metadata.get('total_time_ms', 0)
            ann_time = metadata.get('ann_time_ms', 0)
            rerank_time = metadata.get('rerank_time_ms', 0)
            
            ax_perf = plt.subplot(4, 3, 4)
            metrics = ['ANN', 'Rerank', 'Other']
            values = [ann_time, rerank_time, total_time - ann_time - rerank_time]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            wedges, texts, autotexts = ax_perf.pie(values, labels=metrics, autopct='%1.1f%%',
                                                   colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax_perf.set_title('Time Breakdown')
            
            # Results summary table
            ax_summary = plt.subplot(4, 3, 5)
            ax_summary.axis('off')
            summary_data = [
                ['Metric', 'Value'],
                ['Total Results', str(len(results))],
                ['Total Time', f'{total_time:.0f} ms'],
                ['ANN Time', f'{ann_time:.0f} ms'],
                ['Rerank Time', f'{rerank_time:.0f} ms'],
                ['Avg Distance', f'{np.mean(distances):.2f}'],
                ['Min Distance', f'{min(distances):.2f}'],
                ['Max Distance', f'{max(distances):.2f}']
            ]
            
            table = ax_summary.table(cellText=summary_data, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the header row
            for i in range(len(summary_data[0])):
                table[(0, i)].set_facecolor('#4ecdc4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax_summary.set_title('Search Summary', fontsize=12, fontweight='bold')
            
            # Top results gallery
            ax_gallery = plt.subplot(4, 3, (7, 12))
            ax_gallery.axis('off')
            
            # Create mini result cards for top 6 results
            for i, result in enumerate(results[:6]):
                row = i // 3
                col = i % 3
                x_start = col * 0.33
                y_start = 0.8 - row * 0.25
                
                # Result card
                ax_gallery.text(x_start + 0.165, y_start + 0.2, f"Rank {i+1}: {result['image_id']}", 
                              ha='center', va='center', fontsize=10, fontweight='bold')
                ax_gallery.text(x_start + 0.165, y_start + 0.15, f"Distance: {result['distance']:.2f}", 
                              ha='center', va='center', fontsize=9)
                
                # Color swatches
                colors = result.get('dominant_colors', [])
                for j, color in enumerate(colors[:3]):
                    if x_start + j * 0.05 < x_start + 0.3:
                        rect = patches.Rectangle((x_start + j * 0.05, y_start + 0.05), 0.04, 0.08, 
                                               facecolor=color, edgecolor='black', linewidth=1)
                        ax_gallery.add_patch(rect)
                
                # Border
                border = patches.Rectangle((x_start + 0.01, y_start + 0.01), 0.31, 0.23, 
                                         fill=False, edgecolor='gray', linewidth=1)
                ax_gallery.add_patch(border)
            
            ax_gallery.set_title('Top Results Gallery', fontsize=14, fontweight='bold')
        
        plt.suptitle(f"Comprehensive Search Results Report - {timestamp}", 
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # Save the report
        report_path = output_path / f"search_report_{timestamp}.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive report saved to: {report_path}")
        
        plt.show()

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Search Results Visualization Tool")
    parser.add_argument("--results", type=str, help="Path to JSON file with search results")
    parser.add_argument("--api-query", type=str, help="Query colors (comma-separated hex codes)")
    parser.add_argument("--weights", type=str, default="1.0", help="Color weights (comma-separated)")
    parser.add_argument("--k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--compare", nargs="+", help="Paths to result files for comparison")
    parser.add_argument("--output", type=str, default="search_reports", help="Output directory for reports")
    parser.add_argument("--save", action="store_true", help="Save visualizations to files")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    if not any([args.results, args.api_query, args.compare]):
        parser.print_help()
        return
    
    # Initialize visualizer
    visualizer = SearchResultsVisualizer(api_base_url=args.api_url)
    
    if args.api_query:
        print(f"Querying API with colors: {args.api_query}")
        
        # Parse colors and weights
        colors = [c.strip() for c in args.api_query.split(',')]
        weights = [float(w.strip()) for w in args.weights.split(',')]
        
        # Ensure weights match colors
        if len(weights) == 1 and len(colors) > 1:
            weights = weights * len(colors)
        elif len(weights) != len(colors):
            print("Number of weights must match number of colors")
            return
        
        # Query API
        results_data = visualizer.query_api(colors, weights, args.k)
        
        if results_data and 'results' in results_data:
            results = results_data['results']
            metadata = results_data.get('metadata', {})
            
            print(f"Retrieved {len(results)} results")
            
            # Create visualizations
            visualizer.create_ranking_visualization(results, "API Search Results")
            visualizer.create_performance_analysis(results, metadata, "API Search Performance")
            visualizer.create_result_gallery(results, colors, "API Search Results Gallery")
            
            if args.save:
                output_path = Path(args.output)
                output_path.mkdir(exist_ok=True)
                
                visualizer.create_ranking_visualization(results, "API Search Results",
                                                      output_path / "api_ranking.png")
                visualizer.create_performance_analysis(results, metadata, "API Search Performance",
                                                     output_path / "api_performance.png")
                visualizer.create_result_gallery(results, colors, "API Search Results Gallery",
                                               output_path / "api_gallery.png")
                visualizer.create_comprehensive_report(results, metadata, colors, args.output)
        else:
            print("No results received from API")
    
    elif args.results:
        print(f"Loading results from: {args.results}")
        
        results_data = visualizer.load_results_from_file(args.results)
        
        if results_data and 'results' in results_data:
            results = results_data['results']
            metadata = results_data.get('metadata', {})
            
            print(f"Loaded {len(results)} results")
            
            # Create visualizations
            visualizer.create_ranking_visualization(results, "File Search Results")
            visualizer.create_performance_analysis(results, metadata, "File Search Performance")
            visualizer.create_result_gallery(results, title="File Search Results Gallery")
            
            if args.save:
                output_path = Path(args.output)
                output_path.mkdir(exist_ok=True)
                
                visualizer.create_ranking_visualization(results, "File Search Results",
                                                      output_path / "file_ranking.png")
                visualizer.create_performance_analysis(results, metadata, "File Search Performance",
                                                     output_path / "file_performance.png")
                visualizer.create_result_gallery(results, title="File Search Results Gallery",
                                               save_path=output_path / "file_gallery.png")
                visualizer.create_comprehensive_report(results, metadata, output_dir=args.output)
        else:
            print("No valid results found in file")
    
    elif args.compare:
        print(f"Comparing {len(args.compare)} result files")
        
        # Load and compare results
        all_results = []
        for file_path in args.compare:
            results_data = visualizer.load_results_from_file(file_path)
            if results_data and 'results' in results_data:
                all_results.append((Path(file_path).stem, results_data['results']))
        
        if len(all_results) >= 2:
            # Create comparison visualizations
            for name, results in all_results:
                print(f"Processing {name}: {len(results)} results")
                
                if args.save:
                    output_path = Path(args.output)
                    output_path.mkdir(exist_ok=True)
                    
                    visualizer.create_ranking_visualization(results, f"Results: {name}",
                                                          output_path / f"{name}_ranking.png")
        else:
            print("Need at least 2 valid result files for comparison")

if __name__ == "__main__":
    main()
