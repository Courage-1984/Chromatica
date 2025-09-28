---
created: 2025-09-28T05:30:39 (UTC +02:00)
tags: []
source: https://www.thecolorapi.com/
author: Author
---

# The Color API

> ## Excerpt
> The best color conversion, naming and scheming API out there.

---
___

### Your fast, modern, swiss army knife for color.

Pass in any valid color and get conversion into any other format, the name of the color, placeholder images and a multitude of schemes.

There are only two endpoints you need to worry about, `/id` and `/scheme`, and you can [read the docs](https://www.thecolorapi.com/docs "Documentation") about both. Each endpoint is available in JSON\[P\], HTML & SVG format. The SVG format can be saved or used as an `img[src]` attribute for super-easy embedding/sharing!

Try [Josh's favorite](https://www.thecolorapi.com/id?hex=24B1E0 "Cerulean example"), for example, in [JSON](https://www.thecolorapi.com/id?hex=24B1E0 "Cerulean example"), [HTML](https://www.thecolorapi.com/id?hex=24B1E0&format=html "Cerulean example") or [SVG format](https://www.thecolorapi.com/id?hex=24B1E0&format=svg "Cerulean example").

–––

## How do I convert/identify a color?

All you really need to do is access the `/id` endpoint, and pass in a color value as a query string. [Read the docs](https://www.thecolorapi.com/docs "Documentation") for more details, but all these are valid:

-   `/id?hex=ffa` or `/id?hex=00ffa6`
-   `/id?rgb=rgb(255,0,0)` or `/id?rgb=20,43,55`
-   Same goes for cmyk, hsl, and hsv formats

Every `color` object returned by the API

-   Is named (from a matched dataset of over 2000 names+colors)  
    _e.g. #24B1E0 == Cerulean_
-   Has an image URL for demonstration  
    _e.g. [Cerulean image](https://www.thecolorapi.com/id?hex=24B1E0&format=svg)_
-   Is transposed into hex, rgb, cmyk, hsl, hsv and XYZ formats
-   Is matched to a best-contrast color for text overlay, etc

–––

## How do I generate color schemes?

The parameters are [generally the same](https://www.thecolorapi.com/docs "Documentation") as those necessary for the `/id` endpoint (supply a color, like above), but here you can also specify a scheme mode to guide the generation.

Scheme modes include `monochrome`, `monochrome-dark`, `monochrome-light`, `analogic`, `complement`, `analogic-complement`, `triad` and `quad`.

Every `scheme` object returned by the API is seeded by the color of your request and can be any length you specify (within limits). It will also include a `color` object for each constituent color.

–––

## Anything else?

If you find this [open source API](https://github.com/andjosh/thecolorapi) useful, please [support the developer](https://github.com/sponsors/andjosh)!
