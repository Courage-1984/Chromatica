#!/usr/bin/env python3
"""
Interactive Color Explorer Tool for Chromatica.

This tool provides an interactive interface for exploring color combinations,
analyzing color relationships, and experimenting with different color schemes.

Features:
- Interactive color picker and palette builder
- Color harmony analysis
- Color scheme generation
- Real-time color preview
- Export color palettes

Usage:
    python tools/color_explorer.py
    python tools/color_explorer.py --api-url http://localhost:8000
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider, TextBox
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import requests
import json
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class ColorExplorer:
    """Interactive color exploration and analysis tool."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the color explorer.
        
        Args:
            api_base_url: Base URL for the Chromatica API
        """
        self.api_base_url = api_base_url
        self.current_colors = []
        self.current_weights = []
        self.fig = None
        self.ax_main = None
        self.ax_palette = None
        self.ax_preview = None
        self.ax_controls = None
        
        # Color harmony rules
        self.harmony_rules = {
            'complementary': self._complementary_colors,
            'analogous': self._analogous_colors,
            'triadic': self._triadic_colors,
            'split_complementary': self._split_complementary_colors,
            'tetradic': self._tetradic_colors,
            'monochromatic': self._monochromatic_colors
        }
    
    def create_interface(self):
        """Create the main interactive interface."""
        # Create figure and subplots
        self.fig = plt.figure(figsize=(16, 12))
        
        # Main color display area
        self.ax_main = plt.subplot(2, 3, (1, 4))
        self.ax_main.set_title('Color Explorer', fontsize=16, fontweight='bold')
        self.ax_main.axis('off')
        
        # Palette display
        self.ax_palette = plt.subplot(2, 3, 5)
        self.ax_palette.set_title('Current Palette', fontsize=12, fontweight='bold')
        self.ax_palette.axis('off')
        
        # Preview area
        self.ax_preview = plt.subplot(2, 3, 6)
        self.ax_preview.set_title('Color Preview', fontsize=12, fontweight='bold')
        self.ax_preview.axis('off')
        
        # Controls area
        self.ax_controls = plt.subplot(2, 3, (1, 6))
        self.ax_controls.axis('off')
        
        # Initialize interface
        self._setup_controls()
        self._update_display()
        
        plt.tight_layout()
        plt.show()
    
    def _setup_controls(self):
        """Set up the control elements."""
        # Color input section
        self.ax_controls.text(0.05, 0.9, 'Add Color (Hex):', fontsize=12, fontweight='bold',
                            transform=self.ax_controls.transAxes)
        
        # Color input text box
        self.color_input = TextBox(plt.axes([0.05, 0.85, 0.15, 0.05]), 'Color:')
        self.color_input.on_submit(self._add_color)
        
        # Weight input text box
        self.weight_input = TextBox(plt.axes([0.25, 0.85, 0.1, 0.05]), 'Weight:')
        self.weight_input.text = '1.0'
        
        # Add color button
        self.add_button = Button(plt.axes([0.4, 0.85, 0.1, 0.05]), 'Add')
        self.add_button.on_clicked(self._add_color_from_button)
        
        # Clear palette button
        self.clear_button = Button(plt.axes([0.55, 0.85, 0.1, 0.05]), 'Clear')
        self.clear_button.on_clicked(self._clear_palette)
        
        # Harmony buttons
        self.ax_controls.text(0.05, 0.75, 'Color Harmony:', fontsize=12, fontweight='bold',
                            transform=self.ax_controls.transAxes)
        
        harmony_y = 0.7
        for i, (name, func) in enumerate(self.harmony_rules.items()):
            x_pos = 0.05 + (i % 3) * 0.15
            y_pos = harmony_y - (i // 3) * 0.05
            btn = Button(plt.axes([x_pos, y_pos, 0.12, 0.04]), name.replace('_', ' ').title())
            btn.on_clicked(lambda event, f=func: self._apply_harmony(f))
        
        # Search button
        self.search_button = Button(plt.axes([0.7, 0.85, 0.15, 0.05]), 'Search API')
        self.search_button.on_clicked(self._search_api)
        
        # Export button
        self.export_button = Button(plt.axes([0.7, 0.78, 0.15, 0.05]), 'Export Palette')
        self.export_button.on_clicked(self._export_palette)
        
        # Status text
        self.status_text = self.ax_controls.text(0.05, 0.6, 'Ready to explore colors!', 
                                               fontsize=10, transform=self.ax_controls.transAxes,
                                               color='blue')
    
    def _add_color(self, event):
        """Add a color from the text input."""
        color = event.strip().upper()
        if not color.startswith('#'):
            color = '#' + color
        
        if self._is_valid_hex(color):
            weight = float(self.weight_input.text)
            self._add_color_to_palette(color, weight)
            self.color_input.set_val('')
        else:
            self._update_status(f'Invalid hex color: {color}', 'red')
    
    def _add_color_from_button(self, event):
        """Add color from the add button."""
        color = self.color_input.text.strip().upper()
        if not color.startswith('#'):
            color = '#' + color
        
        if self._is_valid_hex(color):
            weight = float(self.weight_input.text)
            self._add_color_to_palette(color, weight)
            self.color_input.set_val('')
        else:
            self._update_status(f'Invalid hex color: {color}', 'red')
    
    def _add_color_to_palette(self, color: str, weight: float):
        """Add a color to the current palette."""
        self.current_colors.append(color)
        self.current_weights.append(weight)
        self._update_status(f'Added {color} (weight: {weight})', 'green')
        self._update_display()
    
    def _clear_palette(self, event):
        """Clear the current palette."""
        self.current_colors.clear()
        self.current_weights.clear()
        self._update_status('Palette cleared', 'blue')
        self._update_display()
    
    def _apply_harmony(self, harmony_func):
        """Apply a color harmony rule."""
        if not self.current_colors:
            self._update_status('Add a base color first', 'orange')
            return
        
        base_color = self.current_colors[0]
        harmony_colors = harmony_func(base_color)
        
        # Clear current palette and add harmony colors
        self.current_colors = [base_color] + harmony_colors
        self.current_weights = [1.0] * len(self.current_colors)
        
        self._update_status(f'Applied harmony: {len(harmony_colors)} colors added', 'green')
        self._update_display()
    
    def _complementary_colors(self, base_color: str) -> List[str]:
        """Generate complementary colors."""
        rgb = self._hex_to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        
        # Complementary hue (opposite on color wheel)
        comp_hsv = hsv.copy()
        comp_hsv[0] = (hsv[0] + 0.5) % 1.0
        
        comp_rgb = mcolors.hsv_to_rgb(comp_hsv)
        return [self._rgb_to_hex(comp_rgb)]
    
    def _analogous_colors(self, base_color: str) -> List[str]:
        """Generate analogous colors."""
        rgb = self._hex_to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        
        colors = []
        for offset in [-0.083, 0.083]:  # 30 degrees on color wheel
            new_hsv = hsv.copy()
            new_hsv[0] = (hsv[0] + offset) % 1.0
            new_rgb = mcolors.hsv_to_rgb(new_hsv)
            colors.append(self._rgb_to_hex(new_rgb))
        
        return colors
    
    def _triadic_colors(self, base_color: str) -> List[str]:
        """Generate triadic colors."""
        rgb = self._hex_to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        
        colors = []
        for offset in [0.333, 0.667]:  # 120 degrees apart
            new_hsv = hsv.copy()
            new_hsv[0] = (hsv[0] + offset) % 1.0
            new_rgb = mcolors.hsv_to_rgb(new_hsv)
            colors.append(self._rgb_to_hex(new_rgb))
        
        return colors
    
    def _split_complementary_colors(self, base_color: str) -> List[str]:
        """Generate split-complementary colors."""
        rgb = self._hex_to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        
        colors = []
        for offset in [0.417, 0.583]:  # 150 degrees apart
            new_hsv = hsv.copy()
            new_hsv[0] = (hsv[0] + offset) % 1.0
            new_rgb = mcolors.hsv_to_rgb(new_hsv)
            colors.append(self._rgb_to_hex(new_rgb))
        
        return colors
    
    def _tetradic_colors(self, base_color: str) -> List[str]:
        """Generate tetradic colors."""
        rgb = self._hex_to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        
        colors = []
        for offset in [0.25, 0.5, 0.75]:  # 90 degrees apart
            new_hsv = hsv.copy()
            new_hsv[0] = (hsv[0] + offset) % 1.0
            new_rgb = mcolors.hsv_to_rgb(new_hsv)
            colors.append(self._rgb_to_hex(new_rgb))
        
        return colors
    
    def _monochromatic_colors(self, base_color: str) -> List[str]:
        """Generate monochromatic colors."""
        rgb = self._hex_to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        
        colors = []
        for sat_offset in [-0.3, 0.3]:
            for val_offset in [-0.3, 0.3]:
                if sat_offset != 0 or val_offset != 0:
                    new_hsv = hsv.copy()
                    new_hsv[1] = np.clip(hsv[1] + sat_offset, 0, 1)
                    new_hsv[2] = np.clip(hsv[2] + val_offset, 0, 1)
                    new_rgb = mcolors.hsv_to_rgb(new_hsv)
                    colors.append(self._rgb_to_hex(new_rgb))
        
        return colors[:3]  # Limit to 3 additional colors
    
    def _search_api(self, event):
        """Search the Chromatica API with current colors."""
        if not self.current_colors:
            self._update_status('No colors to search with', 'orange')
            return
        
        try:
            # Prepare query
            colors_str = ",".join(self.current_colors)
            weights_str = ",".join(f"{w:.3f}" for w in self.current_weights)
            
            # Make API request
            url = f"{self.api_base_url}/search"
            params = {
                "colors": colors_str,
                "weights": weights_str,
                "k": 5
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            results = response.json()
            
            if results and 'results' in results:
                self._update_status(f'Found {len(results["results"])} similar images', 'green')
                self._show_search_results(results)
            else:
                self._update_status('No results found', 'orange')
                
        except Exception as e:
            self._update_status(f'Search failed: {str(e)}', 'red')
    
    def _show_search_results(self, results_data: Dict):
        """Display search results in a new window."""
        results = results_data['results']
        metadata = results_data.get('metadata', {})
        
        # Create results window
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Search Results', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Display results
        for i, result in enumerate(results):
            x = i % 3
            y = i // 3
            
            # Result card
            card_x = x * 0.33
            card_y = 0.8 - y * 0.25
            
            # Border
            border = patches.Rectangle((card_x + 0.01, card_y + 0.01), 0.31, 0.23, 
                                     fill=False, edgecolor='gray', linewidth=2)
            ax.add_patch(border)
            
            # Content
            ax.text(card_x + 0.165, card_y + 0.2, f"Rank {i+1}: {result['image_id']}", 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(card_x + 0.165, card_y + 0.15, f"Distance: {result['distance']:.2f}", 
                   ha='center', va='center', fontsize=9)
            
            # Color swatches
            colors = result.get('dominant_colors', [])
            for j, color in enumerate(colors[:3]):
                if card_x + j * 0.05 < card_x + 0.3:
                    rect = patches.Rectangle((card_x + j * 0.05, card_y + 0.05), 0.04, 0.08, 
                                           facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
        
        plt.tight_layout()
        plt.show()
    
    def _export_palette(self, event):
        """Export the current palette."""
        if not self.current_colors:
            self._update_status('No colors to export', 'orange')
            return
        
        # Create export data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'colors': self.current_colors,
            'weights': self.current_weights,
            'total_colors': len(self.current_colors)
        }
        
        # Save to file
        output_dir = Path("color_palettes")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"palette_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self._update_status(f'Palette exported to {filename}', 'green')
    
    def _update_display(self):
        """Update the main display area."""
        self.ax_main.clear()
        self.ax_main.axis('off')
        
        if not self.current_colors:
            self.ax_main.text(0.5, 0.5, 'No colors in palette\nAdd colors to get started!', 
                             ha='center', va='center', fontsize=14, 
                             transform=self.ax_main.transAxes)
            return
        
        # Display current colors
        self.ax_main.text(0.5, 0.95, 'Current Color Palette', 
                         ha='center', va='top', fontsize=16, fontweight='bold',
                         transform=self.ax_main.transAxes)
        
        # Create color swatches
        n_colors = len(self.current_colors)
        for i, (color, weight) in enumerate(zip(self.current_colors, self.current_weights)):
            x_pos = 0.1 + i * 0.8 / max(n_colors, 1)
            y_pos = 0.7
            
            # Color swatch
            rect = patches.Rectangle((x_pos, y_pos), 0.6 / max(n_colors, 1), 0.4, 
                                   facecolor=color, edgecolor='black', linewidth=3)
            self.ax_main.add_patch(rect)
            
            # Color info
            self.ax_main.text(x_pos + 0.3 / max(n_colors, 1), y_pos - 0.05, color, 
                             ha='center', va='top', fontsize=10, fontfamily='monospace',
                             transform=self.ax_main.transAxes)
            self.ax_main.text(x_pos + 0.3 / max(n_colors, 1), y_pos - 0.15, f'Weight: {weight}', 
                             ha='center', va='top', fontsize=9,
                             transform=self.ax_main.transAxes)
        
        # Update palette display
        self._update_palette_display()
        
        # Update preview
        self._update_preview()
    
    def _update_palette_display(self):
        """Update the palette display area."""
        self.ax_palette.clear()
        self.ax_palette.axis('off')
        
        if not self.current_colors:
            self.ax_palette.text(0.5, 0.5, 'Empty', ha='center', va='center',
                               transform=self.ax_palette.transAxes)
            return
        
        # Show palette as small swatches
        for i, color in enumerate(self.current_colors):
            x_pos = i * 0.15
            rect = patches.Rectangle((x_pos, 0.1), 0.12, 0.6, 
                                   facecolor=color, edgecolor='black', linewidth=2)
            self.ax_palette.add_patch(rect)
            
            # Color code
            self.ax_palette.text(x_pos + 0.06, 0.05, color, ha='center', va='top',
                               fontsize=8, fontfamily='monospace',
                               transform=self.ax_palette.transAxes)
        
        self.ax_palette.set_xlim(-0.05, len(self.current_colors) * 0.15 + 0.05)
        self.ax_palette.set_ylim(0, 0.8)
    
    def _update_preview(self):
        """Update the preview area."""
        self.ax_preview.clear()
        self.ax_preview.axis('off')
        
        if not self.current_colors:
            self.ax_preview.text(0.5, 0.5, 'No Preview', ha='center', va='center',
                               transform=self.ax_preview.transAxes)
            return
        
        # Create a simple preview pattern
        n_colors = len(self.current_colors)
        
        # Create a grid pattern
        grid_size = int(np.ceil(np.sqrt(n_colors)))
        for i, color in enumerate(self.current_colors):
            row = i // grid_size
            col = i % grid_size
            
            x_pos = col * 0.8 / grid_size + 0.1
            y_pos = 0.8 - row * 0.6 / grid_size
            
            rect = patches.Rectangle((x_pos, y_pos), 0.7 / grid_size, 0.5 / grid_size, 
                                   facecolor=color, edgecolor='white', linewidth=2)
            self.ax_preview.add_patch(rect)
    
    def _update_status(self, message: str, color: str = 'black'):
        """Update the status message."""
        self.status_text.set_text(message)
        self.status_text.set_color(color)
        self.fig.canvas.draw_idle()
    
    def _is_valid_hex(self, color: str) -> bool:
        """Check if a string is a valid hex color."""
        if not color.startswith('#'):
            return False
        
        if len(color) not in [4, 7]:  # #RGB or #RRGGBB
            return False
        
        try:
            int(color[1:], 16)
            return True
        except ValueError:
            return False
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB tuple (0-1)."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c + c for c in hex_color])
        
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return tuple(c / 255.0 for c in rgb)
    
    def _rgb_to_hex(self, rgb: Tuple[float, float, float]) -> str:
        """Convert RGB tuple (0-1) to hex color."""
        rgb_255 = tuple(int(c * 255) for c in rgb)
        return f"#{rgb_255[0]:02x}{rgb_255[1]:02x}{rgb_255[2]:02x}"

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Interactive Color Explorer Tool")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", 
                       help="API base URL")
    
    args = parser.parse_args()
    
    try:
        # Try to print with emoji, fallback to text if encoding fails
        print("🎨 Chromatica Color Explorer")
        print("=" * 40)
        print("Features:")
        print("- Interactive color picker")
        print("- Color harmony generation")
        print("- Palette building and export")
        print("- API search integration")
        print("=" * 40)
    except UnicodeEncodeError:
        # Fallback for systems that can't handle emoji
        print("Chromatica Color Explorer")
        print("=" * 40)
        print("Features:")
        print("- Interactive color picker")
        print("- Color harmony generation")
        print("- Palette building and export")
        print("- API search integration")
        print("=" * 40)
    
    # Check if we're in an interactive environment
    try:
        import matplotlib
        backend = matplotlib.get_backend()
        is_interactive = matplotlib.is_interactive()
        
        if not is_interactive or 'Agg' in backend:
            # Non-interactive environment - run demo mode
            print("\nRunning in non-interactive mode - showing demo capabilities:")
            print("=" * 50)
            
            # Create explorer for demo
            explorer = ColorExplorer(api_base_url=args.api_url)
            
            # Demo color harmonies
            demo_colors = ['#FF0000', '#00FF00', '#0000FF']
            print(f"\nDemo Colors: {demo_colors}")
            
            # Show complementary colors
            comp_colors = explorer._complementary_colors(demo_colors[0])
            print(f"Complementary to {demo_colors[0]}: {comp_colors}")
            
            # Show analogous colors
            anal_colors = explorer._analogous_colors(demo_colors[0])
            print(f"Analogous to {demo_colors[0]}: {anal_colors}")
            
            # Show triadic colors
            tri_colors = explorer._triadic_colors(demo_colors[0])
            print(f"Triadic to {demo_colors[0]}: {tri_colors}")
            
            print("\nDemo completed successfully!")
            print("For full interactive features, run this tool in an interactive environment.")
            
        else:
            # Interactive environment - create full interface
            print("\nCreating interactive interface...")
            explorer = ColorExplorer(api_base_url=args.api_url)
            explorer.create_interface()
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Running in fallback demo mode...")
        
        # Fallback demo
        print("\nDemo Color Explorer Features:")
        print("- Color harmony generation")
        print("- Palette analysis")
        print("- API integration capabilities")
        print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
