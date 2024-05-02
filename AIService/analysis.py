import cv2
import tensorflow as tf

def preprocess_image(image):
    # Resize and normalize the image for MobileNetV2
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    # Add a batch dimension
    image = tf.expand_dims(image, axis=0)
    return image

# List of recyclable items
recyclable_items = ['bottle', 'cardboard', 'paper', 'can','plastic_bag']

def detect_objects(image):
    # Load a pre-trained TensorFlow model
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
    # Use the model to predict the objects in the image
    predictions = model.predict(image)
    # For simplicity, let's just return the top 5 predictions
    top_5 = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    # Convert the predictions to a list of detections
    detections = [(label, score) for _, label, score in top_5]
    return detections

treatment_plans = {
    'bottle': 'Clean and place in recycling bin.',
    'cardboard': 'Flatten and place in recycling bin.',
    'paper': 'Place in paper recycling bin.',
    'can': 'Rinse and place in metal recycling bin.',
    'plastic_bag': 'Reuse if possible, otherwise place in plastic recycling bin.'
}

def propose_treatment_plan(item):
    # Look up the treatment plan for the item
    plan = treatment_plans.get(item)
    if plan:
        print(f'Treatment plan for {item}: {plan}')
    else:
        print(f'No treatment plan found for {item}.')

# Load and preprocess the input image
image = cv2.imread('E:/THT_3/AIService/t1.jpg')
processed_image = preprocess_image(image)

# Perform object detection
detections = detect_objects(processed_image)

# Check if the detected objects are recyclable
recyclable_detections = [label for label, score in detections if label in recyclable_items]

if recyclable_detections:
    print('Detected recyclable items:', recyclable_detections)
else:
    print('No recyclable items detected.')

for label, score in detections:
    if label in recyclable_items:
        propose_treatment_plan(label)