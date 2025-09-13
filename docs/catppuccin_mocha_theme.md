# 🎨 Catppuccin Mocha Theme Documentation

## Overview

This document details the implementation of the Catppuccin Mocha theme in the Chromatica Color Search Engine web interface. The theme provides a soothing, dark pastel aesthetic that enhances user experience while maintaining excellent readability and accessibility.

## 🎯 Theme Philosophy

The Catppuccin Mocha theme follows the design principles of:

- **Soothing Pastels**: Gentle, eye-friendly colors that reduce eye strain
- **High Contrast**: Excellent readability in dark environments
- **Consistent Aesthetics**: Unified color scheme across all interface elements
- **Accessibility**: WCAG-compliant color combinations

## 🌈 Color Palette

### Base Colors

```css
--base: #1e1e2e      /* Primary background */
--mantle: #181825    /* Secondary background */
--crust: #11111b     /* Container background */
```

### Surface Colors

```css
--surface0: #313244  /* Primary surface */
--surface1: #45475a  /* Secondary surface */
--surface2: #585b70  /* Tertiary surface */
```

### Overlay Colors

```css
--overlay0: #6c7086  /* Primary overlay */
--overlay1: #7f849c  /* Secondary overlay */
--overlay2: #9399b2  /* Tertiary overlay */
```

### Text Colors

```css
--text: #cdd6f4      /* Primary text */
--subtext0: #a6adc8  /* Secondary text */
--subtext1: #bac2de  /* Tertiary text */
```

### Accent Colors

```css
--rosewater: #f5e0dc /* Warm accent */
--flamingo: #f2cdcd  /* Soft pink */
--pink: #f5c2e7     /* Vibrant pink */
--mauve: #cba6f7    /* Purple accent */
--red: #f38ba8      /* Error/remove */
--maroon: #eba0ac    /* Dark red */
--peach: #fab387     /* Orange accent */
--yellow: #f9e2af    /* Warm yellow */
--green: #a6e3a1     /* Success/primary */
--teal: #94e2d5      /* Teal accent */
--sky: #89dceb       /* Light blue */
--sapphire: #74c7ec  /* Blue accent */
--blue: #89b4fa      /* Primary blue */
--lavender: #b4befe  /* Light purple */
```

## 🎨 Implementation Details

### CSS Custom Properties

The theme uses CSS custom properties (CSS variables) for consistent color application:

```css
:root {
  /* All color definitions */
}
```

### Color Application Strategy

#### Backgrounds

- **Body**: Gradient from `--base` to `--mantle`
- **Container**: `--crust` with `--surface0` border
- **Sections**: `--surface0` with `--surface1` borders
- **Cards**: `--surface0` backgrounds

#### Text

- **Headings**: `--text` for maximum contrast
- **Body text**: `--text` for primary content
- **Secondary text**: `--subtext1` for descriptions
- **Muted text**: `--subtext0` for labels

#### Interactive Elements

- **Primary buttons**: `--blue` with `--crust` text
- **Success buttons**: `--green` with `--crust` text
- **Secondary buttons**: `--peach` with `--crust` text
- **Info buttons**: `--mauve` with `--crust` text
- **Remove buttons**: `--red` with `--crust` text

#### Hover States

- **Cards**: `--mauve` border on hover
- **Buttons**: Darker variants of base colors
- **Interactive elements**: Subtle color shifts

## 🔧 Technical Implementation

### File Location

```
src/chromatica/api/static/index.html
```

### CSS Structure

The theme implementation follows this structure:

1. **CSS Variables Declaration** (lines 7-42)
2. **Base Element Styling** (body, container, headings)
3. **Component-Specific Styling** (cards, buttons, forms)
4. **Interactive State Styling** (hover, focus, disabled)
5. **Responsive Design Considerations**

### Key CSS Classes

#### Container Classes

- `.container`: Main page container with dark background
- `.color-input-section`: Color picker interface
- `.results-section`: Search results display
- `.visualization-section`: Data visualization area
- `.visualization-tools-section`: Tool selection interface

#### Card Classes

- `.result-card`: Individual search result cards
- `.tool-card`: Visualization tool selection cards
- `.viz-card`: Visualization display cards

#### Button Classes

- `.search-btn`: Primary search button
- `.add-color-btn`: Add color button
- `.action-btn`: Result action buttons
- `.tool-btn`: Tool interface buttons

## 🎭 Visual Elements

### Color Input Interface

- **Color pickers**: Standard HTML color inputs with enhanced styling
- **Weight sliders**: Custom-styled range inputs with mauve thumbs
- **Add/Remove buttons**: Clear visual hierarchy with appropriate colors

### Search Results

- **Result cards**: Dark surfaces with subtle borders
- **Image displays**: Bordered image containers
- **Color swatches**: Interactive color previews
- **Action buttons**: Consistent button styling

### Visualization Tools

- **Tool icons**: Gradient backgrounds using mauve and pink
- **Feature lists**: Checkmark indicators in green
- **Category sections**: Subtle surface variations

## 📱 Responsive Design

### Mobile Considerations

- **Grid layouts**: Responsive grid systems
- **Touch targets**: Appropriate button sizes
- **Color contrast**: Maintained across screen sizes

### Breakpoint Strategy

```css
@media (max-width: 768px) {
  .container {
    padding: 20px;
  }
  .tools-grid {
    grid-template-columns: 1fr;
  }
  .results-grid {
    grid-template-columns: 1fr;
  }
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
```

## ♿ Accessibility Features

### Color Contrast

- **Text on dark backgrounds**: Minimum 4.5:1 contrast ratio
- **Interactive elements**: Clear visual feedback
- **Error states**: Distinct red coloring for alerts

### Visual Hierarchy

- **Consistent spacing**: Uniform margins and padding
- **Clear borders**: Subtle but visible element separation
- **Hover states**: Obvious interactive feedback

### Screen Reader Support

- **Semantic HTML**: Proper heading structure
- **Alt text**: Image descriptions maintained
- **ARIA labels**: Interactive element labeling

## 🚀 Usage Guidelines

### Adding New Components

When adding new UI components, follow these guidelines:

1. **Use CSS variables**: Reference the defined color palette
2. **Maintain contrast**: Ensure text readability
3. **Follow patterns**: Use established component styles
4. **Test accessibility**: Verify color contrast ratios

### Color Selection

- **Primary actions**: Use `--blue` or `--green`
- **Secondary actions**: Use `--peach` or `--mauve`
- **Destructive actions**: Use `--red`
- **Information**: Use `--mauve` or `--blue`

### Hover States

- **Cards**: Add `--mauve` border
- **Buttons**: Use darker variants
- **Interactive elements**: Subtle color shifts

## 🔍 Customization Options

### Theme Variations

The CSS variable system allows easy theme customization:

```css
/* Light theme example */
:root {
  --base: #eff1f5;
  --text: #4c4f69;
  /* ... other light colors */
}
```

### Color Adjustments

Individual colors can be modified by changing CSS variables:

```css
:root {
  --primary: #custom-color;
  --accent: #custom-accent;
}
```

## 📊 Performance Considerations

### CSS Optimization

- **CSS variables**: Efficient color management
- **Minimal repaints**: Optimized hover states
- **Responsive design**: Efficient media queries

### Browser Support

- **Modern browsers**: Full CSS variable support
- **Fallbacks**: Graceful degradation for older browsers
- **Progressive enhancement**: Core functionality maintained

## 🧪 Testing and Validation

### Visual Testing

- **Color accuracy**: Verify against Catppuccin palette
- **Contrast ratios**: Validate accessibility standards
- **Cross-browser**: Test across different browsers

### Accessibility Testing

- **Screen readers**: Verify navigation and content
- **Keyboard navigation**: Test tab order and focus
- **Color blindness**: Ensure sufficient contrast

## 📚 References

### Official Resources

- [Catppuccin Official Website](https://catppuccin.com/)
- [Catppuccin GitHub Repository](https://github.com/catppuccin/catppuccin)
- [Catppuccin Palette Documentation](https://github.com/catppuccin/palette)

### Design Principles

- **Dark Theme Best Practices**: Accessibility guidelines
- **Color Theory**: Understanding color relationships
- **UI/UX Design**: Modern interface design principles

## 🔄 Maintenance

### Regular Updates

- **Color palette**: Keep synchronized with official releases
- **Accessibility**: Regular contrast ratio testing
- **Browser compatibility**: Test with new browser versions

### Version Control

- **CSS changes**: Document all color modifications
- **Theme updates**: Track palette version changes
- **Testing results**: Maintain accessibility validation records

---

## 📝 Changelog

### Version 1.0.0 (Current)

- **Initial implementation**: Complete Catppuccin Mocha theme
- **Color palette**: Full 25-color implementation
- **Component styling**: All interface elements themed
- **Accessibility**: WCAG-compliant contrast ratios
- **Responsive design**: Mobile-optimized layouts

---

_This documentation is maintained as part of the Chromatica project and should be updated whenever theme changes are made._
