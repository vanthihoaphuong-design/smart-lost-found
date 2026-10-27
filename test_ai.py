from ai.detect import detect_object

name, score = detect_object("uploads/test.jpg")

print(name)

print(score)