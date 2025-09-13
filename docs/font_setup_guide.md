# Font Setup Guide for Chromatica Web Interface

## Overview
The Chromatica web interface uses custom fonts to provide a consistent and professional appearance that matches the Catppuccin Mocha theme requirements. This guide covers font installation, configuration, and troubleshooting.

## Font Requirements
- **Primary Font**: JetBrains Mono Nerd Font Mono (for code and UI elements)
- **Emoji Font**: Segoe UI Emoji (for emoji support)
- **Symbol Font**: Segoe UI Symbol (for special symbols)

## Font Files Location
Font files are stored in the following directory structure:
```
src/chromatica/api/static/fonts/
├── JetBrainsMonoNerdFontMono-Regular.ttf
├── JetBrainsMonoNerdFontMono-Bold.ttf
├── JetBrainsMonoNerdFontMono-Italic.ttf
├── JetBrainsMonoNerdFontMono-BoldItalic.ttf
├── JetBrainsMonoNerdFontMono-Medium.ttf
├── JetBrainsMonoNerdFontMono-SemiBold.ttf
├── seguiemj.ttf
└── seguisym.ttf
```

## Font Serving Configuration
The FastAPI application serves static files from the `src/chromatica/api/static/` directory and mounts them at the `/static` endpoint. This means:

- **Font URLs**: Fonts are accessible at `/static/fonts/[filename]`
- **CSS References**: All font-face declarations use absolute paths starting with `/static/fonts/`

## CSS Font-Face Declarations
The web interface includes comprehensive font-face declarations for all font weights and styles:

```css
@font-face {
    font-family: 'JetBrainsMono Nerd Font Mono';
    src: url('/static/fonts/JetBrainsMonoNerdFontMono-Regular.ttf') format('truetype');
    font-weight: 400;
    font-style: normal;
}

@font-face {
    font-family: 'JetBrainsMono Nerd Font Mono';
    src: url('/static/fonts/JetBrainsMonoNerdFontMono-Bold.ttf') format('truetype');
    font-weight: 700;
    font-style: normal;
}

/* Additional weights and styles... */
```

## Troubleshooting Font Issues

### Common Issues and Solutions

#### 1. Font Files Not Loading (404 Errors)
**Symptoms**: Browser console shows 404 errors for font files
**Cause**: Incorrect font paths in CSS or missing font files
**Solution**: 
- Verify font files exist in `src/chromatica/api/static/fonts/`
- Ensure CSS uses absolute paths starting with `/static/fonts/`
- Check that FastAPI static file mounting is working

#### 2. Fonts Not Applied
**Symptoms**: Text appears in fallback fonts instead of custom fonts
**Cause**: Font loading failures or CSS specificity issues
**Solution**:
- Check browser console for font loading errors
- Verify font-family declarations in CSS
- Ensure font files are valid TTF files

#### 3. Missing Font Weights
**Symptoms**: Some text appears bold when it should be regular, or vice versa
**Cause**: Missing font files for specific weights
**Solution**:
- Verify all required font weight files are present
- Check CSS font-weight declarations match available files

### Debugging Steps

1. **Check Browser Console**: Look for 404 errors or font loading failures
2. **Verify File Paths**: Confirm fonts are accessible at `/static/fonts/[filename]`
3. **Test Font Loading**: Use browser dev tools to check if fonts are loaded
4. **Check CSS**: Verify font-face declarations use correct paths

### Font Loading Verification
To verify fonts are loading correctly:

1. Open browser dev tools (F12)
2. Go to Network tab
3. Refresh the page
4. Look for successful font file requests (should show 200 OK)
5. Check that font URLs match `/static/fonts/[filename]` pattern

## Font Performance Optimization

### Font Loading Strategy
- **Preload**: Critical fonts are loaded early for optimal performance
- **Fallbacks**: Comprehensive fallback font stacks ensure text remains readable
- **Subset**: Fonts include only necessary characters to reduce file size

### Browser Compatibility
- **Modern Browsers**: Full support for TTF fonts and font-face declarations
- **Fallback Support**: Graceful degradation to system fonts if custom fonts fail to load

## Maintenance and Updates

### Adding New Fonts
1. Place font files in `src/chromatica/api/static/fonts/`
2. Add corresponding `@font-face` declarations in the CSS
3. Update font-family declarations as needed
4. Test font loading across different browsers

### Font File Management
- Keep font files organized in the dedicated fonts directory
- Use descriptive filenames that indicate weight and style
- Regularly verify font file integrity and loading

## Related Documentation
- [Catppuccin Mocha Theme Guide](catppuccin_mocha_theme.md)
- [Web Interface Configuration](visualization_features.md)
- [API Configuration](api_reference.md)

## Support and Troubleshooting
For additional font-related issues, refer to:
- [Troubleshooting Guide](troubleshooting.md)
- [API Logs](logs/) for server-side font serving issues
- Browser developer tools for client-side font loading problems
