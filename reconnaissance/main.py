import os
import cv2#load images
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

#load mnist
mnist =  tf.keras.datasets.mnist

#split into training data 
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test =  tf.keras.utils.normalize(x_test, axis=1)

#create the model 
model = tf.keras.models.Sequential()

#add layers 
#starting with flatten them 
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

#dense layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train model 
model.fit(x_train, y_train, epochs=3)

model.save('handwritten.keras')

#once trained, when can call it 
#model = tf.keras.models.load_model('handwritten.keras')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

image_number = 1
while True:
    image_path = f"digits/digit{image_number}.png"
    if not os.path.isfile(image_path):
        break
    
    try:
        print("Reading image:", image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Print the input image
        plt.imshow(img, cmap='gray')
        plt.title('original Image')
        plt.show()
        
        img_resized = cv2.resize(img, (28, 28))
        img_normalized = np.invert(img_resized) / 255.0
        
        # Print the preprocessed image
        plt.imshow(img_normalized, cmap='gray')
        plt.title('mnist version Image')
        plt.show()
        
        # Reshape the image for prediction
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # Get the model's prediction probabilities
        prediction_probs = model.predict(img_input)[0]
        
        # Print the prediction probabilities
        print("Prediction Probabilities:")
        for digit, prob in enumerate(prediction_probs):
            print(f"Digit {digit}: Probability {prob:.6f}")
        
        # Get the predicted digit
        predicted_digit = np.argmax(prediction_probs)
        print(f"This digit is probably a {predicted_digit}")
        
        # Show the inverted image
        img_resized_inverted = cv2.bitwise_not(img_resized)
        plt.imshow(img_resized_inverted, cmap='gray')
        plt.title('the prediction')
        plt.show()
        
    except Exception as e:
        print("ERROR:", e)
    finally:
        image_number += 1  

    image_path = f"digits/digit{image_number}.png"
    if not os.path.isfile(image_path):
        break
    
    try:
        print("Reading image:", image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Print the image to inspect visually
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.show()
        
        img_resized = cv2.resize(img, (28, 28))
        img_normalized = np.invert(img_resized) / 255.0
        
        # Print the preprocessed image to inspect visually
        plt.imshow(img_normalized, cmap='gray')
        plt.title('Preprocessed Image')
        plt.show()
        
        prediction = model.predict(np.array([img_normalized]))
        predicted_digit = np.argmax(prediction)
        print(f"This digit is probably a {predicted_digit}")
        
        img_resized_inverted = cv2.bitwise_not(img_resized)
        plt.imshow(img_resized_inverted, cmap='gray')
        plt.title('Inverted Image')
        plt.show()
        
    except Exception as e:
        print("ERROR:", e)
    finally:
        image_number += 1  
