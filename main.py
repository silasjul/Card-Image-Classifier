from model import predict_image, test_model, train_model, SimpleCardClassifier
from glob import glob

# Model training and testing example
model_path = "playing_card_classifier.pt"
model = SimpleCardClassifier(num_classes=53) # 52 cards + joker

#train_model(model_path, epochs=5)
#test_model(model_path, visualize_wrong_predictions=False)

# Single image prediction example
predicted_class, confidence = predict_image(model_path, image_path="./dataset/predict/1.png", visualize_result=False)
print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")

# Multiple image prediction example
images = glob("./dataset/predict/*")
for image in images:
    predicted_class, confidence = predict_image(model_path, image_path=image, visualize_result=True)