from PIL import Image

# Open the images
foreground = Image.open('/Users/dipashrestha/Downloads/aa_re.png').convert("RGBA")
background = Image.open('/Users/dipashrestha/Downloads/tu.png').convert("RGBA")

# Resize background to match the size of the foreground
background = background.resize(foreground.size, Image.ANTIALIAS)

# Combine the images
combined = Image.alpha_composite(background, foreground)

# Save the output
combined.save('/Users/dipashrestha/Downloads/aa_combine.png')

print("Background replaced successfully.")
