# Assets Folder

This folder contains images and assets for the project documentation.

## Banner Image

Place your combined banner image here as `banner.png`.

The banner should combine:
1. Landing page view (light theme) showing:
   - TubeScribe AI logo and title
   - Feature buttons (Transcribe, Translate, Summarize)
   - "Start Analyzing" button
   - Chatbot interface on the right

2. Main application view (dark theme) showing:
   - Application interface
   - Processing features
   - Chat interface

### Recommended Specifications:
- **Dimensions**: 1200x600px (or 2:1 aspect ratio)
- **Format**: PNG or JPG
- **File name**: `banner.png`

### How to Create Combined Banner:

1. **Using Image Editor (Photoshop/GIMP/Canva)**:
   - Create a 1200x600px canvas
   - Place both images side-by-side or stack them vertically
   - Add any connecting elements or transitions
   - Export as PNG

2. **Using Online Tools**:
   - Use tools like Photopea (free Photoshop alternative)
   - Or use Canva/Figma to combine the images
   - Save as PNG with transparent or solid background

3. **Using Command Line** (ImageMagick):
   ```bash
   # Combine horizontally
   convert image1.png image2.png +append banner.png
   
   # Or combine vertically
   convert image1.png image2.png -append banner.png
   ```

Once the image is created, place it in this folder as `banner.png`.
