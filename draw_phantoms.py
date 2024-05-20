from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math

big_circle_center = (107, 107)
big_circle_radius = 105 
#small_circle_center = (110, 110)
small_circle_radius = 20  
y = 107
big_circle_color = (100, 100, 100)  # gray
small_circle_color = (200, 200, 200)  # lighter gray


def draw_all_in_one(big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color, title, y):
    image_size = (214, 214)
    image = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(image)

    draw.ellipse((big_circle_center[0] - big_circle_radius, 
                big_circle_center[1] - big_circle_radius,
                big_circle_center[0] + big_circle_radius, 
                big_circle_center[1] + big_circle_radius), 
                fill=big_circle_color)

    x = 2+small_circle_radius+4
    draw.ellipse((x - small_circle_radius, y - small_circle_radius, x + small_circle_radius, y + small_circle_radius), fill=small_circle_color)
    draw.ellipse((y - small_circle_radius, x - small_circle_radius, y + small_circle_radius, x + small_circle_radius), fill=small_circle_color)

    x = (2+small_circle_radius+4)+small_circle_radius*2
    draw.ellipse((x - small_circle_radius, y - small_circle_radius, x + small_circle_radius, y + small_circle_radius), fill=small_circle_color)
    draw.ellipse((y - small_circle_radius, x - small_circle_radius, y + small_circle_radius, x + small_circle_radius), fill=small_circle_color)
    y1 = y+small_circle_radius*2
    draw.ellipse((x - small_circle_radius, y1 - small_circle_radius, x + small_circle_radius, y1 + small_circle_radius), fill=small_circle_color)
    y1 = y-small_circle_radius*2
    draw.ellipse((x - small_circle_radius, y1 - small_circle_radius, x + small_circle_radius, y1 + small_circle_radius), fill=small_circle_color)

    x = big_circle_center[0]
    draw.ellipse((x - small_circle_radius, y - small_circle_radius, x + small_circle_radius, y + small_circle_radius), fill=small_circle_color)
    draw.ellipse((y - small_circle_radius, x - small_circle_radius, y + small_circle_radius, x + small_circle_radius), fill=small_circle_color)

    x = (212-small_circle_radius-4)-small_circle_radius*2
    draw.ellipse((x - small_circle_radius, y - small_circle_radius, x + small_circle_radius, y + small_circle_radius), fill=small_circle_color)
    draw.ellipse((y - small_circle_radius, x - small_circle_radius, y + small_circle_radius, x + small_circle_radius), fill=small_circle_color)
    y1 = y+small_circle_radius*2
    draw.ellipse((x - small_circle_radius, y1 - small_circle_radius, x + small_circle_radius, y1 + small_circle_radius), fill=small_circle_color)
    y1 = y-small_circle_radius*2
    draw.ellipse((x - small_circle_radius, y1 - small_circle_radius, x + small_circle_radius, y1 + small_circle_radius), fill=small_circle_color)

    x = 212-small_circle_radius-4
    draw.ellipse((x - small_circle_radius, y - small_circle_radius, x + small_circle_radius, y + small_circle_radius), fill=small_circle_color)
    draw.ellipse((y - small_circle_radius, x - small_circle_radius, y + small_circle_radius, x + small_circle_radius), fill=small_circle_color)
    image = image.convert("L")
    # Save the image
    image.save(f"draw/{title}.png")
    image.save(f"draw/{title}.jpg")
    # Show the image (optional)
    image.close()

    return 0

def draw(x,y,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color, title):
    image_size = (214, 214)
    image = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(image)

    draw.ellipse((big_circle_center[0] - big_circle_radius, 
                big_circle_center[1] - big_circle_radius,
                big_circle_center[0] + big_circle_radius, 
                big_circle_center[1] + big_circle_radius), 
                fill=big_circle_color)

    draw.ellipse((x - small_circle_radius, y - small_circle_radius, x + small_circle_radius, y + small_circle_radius), fill=small_circle_color)
    #draw.ellipse((x , y , x + 2*small_circle_radius, y + 2*small_circle_radius), fill=small_circle_color)
    
    # Save the image
    image = image.convert("L")
    image.save(f"draw/{title}.png")
    image.save(f"draw/{title}.jpg")
    image.close()

    return 0


x = 2+small_circle_radius+4
draw(x,y,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y}') # f'x-{y}_y-{x}'
draw(y,x,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{y}_y-{x}')

x = (2+small_circle_radius+4)+small_circle_radius*2
draw(y,x,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{y}_y-{x}')
draw(x,y,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y}')
y1 = y+small_circle_radius*2
draw(x,y1,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y1}')
y1 = y-small_circle_radius*2
draw(x,y1,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y1}')

x = big_circle_center[0]
draw(y,x,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{y}_y-{x}')
draw(x,y,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y}')

x = (212-small_circle_radius-4)-small_circle_radius*2
draw(x,y,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y}')
draw(y,x,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{y}_y-{x}')
y1 = y+small_circle_radius*2
draw(x,y1,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y1}')
y1 = y-small_circle_radius*2
draw(x,y1,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y1}')

x = 212-small_circle_radius-4
draw(y,x,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{y}_y-{x}')
draw(x,y,big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,f'x-{x}_y-{y}')

draw_all_in_one(big_circle_center,big_circle_radius,small_circle_radius, big_circle_color, small_circle_color,"all_in_one", y)

print("Done")

for x in range(0, big_circle_center[0]*2, small_circle_radius):
    for y in range(0, big_circle_center[1]*2, small_circle_radius):
        #Calculate the distance from the center of the smaller circle to the current point
        distance = math.sqrt(x * x + y * y)
        if distance <= small_circle_radius*2:
        draw(x,y,big_circle_center,big_circle_radius,small_circle_radius, f'x-{x}_y-{y}')
