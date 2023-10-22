from tools4eyecolor import tools

detect = tools()
b64 = detect.to_image_string("cropped/o56.jpg")
color_detected = detect.eye_color(b64)
print("Detected Iris Color: " + str(color_detected))