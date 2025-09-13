# 🎨 Catppuccin Mocha Theme - Quick Reference

## 🚀 Quick Start

### Basic Usage

```css
/* Use CSS variables for consistent theming */
.my-component {
  background: var(--surface0);
  color: var(--text);
  border: 1px solid var(--surface1);
}
```

### Common Patterns

```css
/* Cards */
.card {
  background: var(--surface0);
  border: 2px solid var(--surface1);
  border-radius: 12px;
}

/* Buttons */
.btn-primary {
  background: var(--blue);
  color: var(--crust);
}
.btn-success {
  background: var(--green);
  color: var(--crust);
}
.btn-secondary {
  background: var(--peach);
  color: var(--crust);
}
.btn-danger {
  background: var(--red);
  color: var(--crust);
}

/* Hover effects */
.interactive:hover {
  border-color: var(--mauve);
  transform: translateY(-2px);
}
```

## 🌈 Color Quick Reference

### Backgrounds

| Use Case        | Variable     | Hex       |
| --------------- | ------------ | --------- |
| Main background | `--base`     | `#1e1e2e` |
| Container       | `--crust`    | `#11111b` |
| Cards/sections  | `--surface0` | `#313244` |
| Borders         | `--surface1` | `#45475a` |

### Text

| Use Case  | Variable     | Hex       |
| --------- | ------------ | --------- |
| Headings  | `--text`     | `#cdd6f4` |
| Body text | `--text`     | `#cdd6f4` |
| Secondary | `--subtext1` | `#bac2de` |
| Muted     | `--subtext0` | `#a6adc8` |

### Actions

| Use Case | Variable  | Hex       |
| -------- | --------- | --------- |
| Primary  | `--blue`  | `#89b4fa` |
| Success  | `--green` | `#a6e3a1` |
| Warning  | `--peach` | `#fab387` |
| Danger   | `--red`   | `#f38ba8` |
| Info     | `--mauve` | `#cba6f7` |

## 🎯 Component Examples

### Form Elements

```css
.input-field {
  background: var(--surface0);
  border: 2px solid var(--surface1);
  color: var(--text);
  border-radius: 8px;
}

.input-field:focus {
  border-color: var(--mauve);
  outline: none;
}
```

### Navigation

```css
.nav-item {
  color: var(--subtext1);
  transition: color 0.2s;
}

.nav-item:hover,
.nav-item.active {
  color: var(--text);
}
```

### Status Indicators

```css
.status-success {
  color: var(--green);
}
.status-warning {
  color: var(--peach);
}
.status-error {
  color: var(--red);
}
.status-info {
  color: var(--mauve);
}
```

## 📱 Responsive Considerations

### Mobile Breakpoints

```css
@media (max-width: 768px) {
  .container {
    padding: 20px;
  }
  .grid {
    grid-template-columns: 1fr;
  }
}
```

### Touch Targets

```css
/* Ensure minimum 44px touch targets */
.touch-target {
  min-height: 44px;
  min-width: 44px;
}
```

## ♿ Accessibility Guidelines

### Contrast Ratios

- **Normal text**: Minimum 4.5:1
- **Large text**: Minimum 3:1
- **UI components**: Minimum 3:1

### Focus States

```css
.focusable:focus {
  outline: 2px solid var(--mauve);
  outline-offset: 2px;
}
```

### Screen Reader Support

```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
```

## 🔧 Development Workflow

### Adding New Components

1. **Use existing variables** - Don't create new colors
2. **Follow established patterns** - Match existing component styles
3. **Test contrast** - Verify accessibility compliance
4. **Document changes** - Update theme documentation

### Testing Checklist

- [ ] Colors match Catppuccin palette
- [ ] Contrast ratios meet WCAG standards
- [ ] Hover states provide clear feedback
- [ ] Mobile layout works correctly
- [ ] Screen reader compatibility verified

### Common Mistakes

❌ **Don't use hardcoded colors**

```css
/* Wrong */
.my-component {
  color: #ffffff;
}

/* Right */
.my-component {
  color: var(--text);
}
```

❌ **Don't ignore hover states**

```css
/* Wrong */
.button {
  background: var(--blue);
}

/* Right */
.button {
  background: var(--blue);
  transition: background 0.2s;
}
.button:hover {
  background: var(--sapphire);
}
```

❌ **Don't forget mobile**

```css
/* Wrong */
.grid {
  grid-template-columns: repeat(3, 1fr);
}

/* Right */
.grid {
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}
```

## 📚 Resources

### Official Documentation

- [Full Theme Documentation](./catppuccin_mocha_theme.md)
- [Catppuccin Website](https://catppuccin.com/)
- [Color Palette Reference](https://github.com/catppuccin/palette)

### Tools

- [Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [Color Blindness Simulator](https://www.toptal.com/designers/colorfilter)
- [CSS Variable Support](https://caniuse.com/css-variables)

---

_This quick reference is part of the Chromatica project documentation. For detailed information, see the full theme documentation._
