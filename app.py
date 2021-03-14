import cv2
import os
import numpy as np
# import sklearn
# import sklearn.model_selection
import keras
from keras.models import load_model



# datafolder='SomethingOrNothing'
# data=[]
# labels=[]
# folders=['Something','Nothing']
# for category in folders:
#     path = os.path.join(datafolder,category)
#     images = os.listdir(path)
#     for eachImage in images:
#         imgarray = cv2.imread(os.path.join(path,eachImage)) 
#         data.append(imgarray)
#         if category == "Something":
#             labels.append(0)
#         if category == "Nothing":
#             labels.append(1)

# data = np.array(data)
# labels = np.array(labels)


# train_images,test_images,train_labels,test_labels=sklearn.model_selection.train_test_split(data,labels,test_size=0.1)
# train_images=train_images/255
# test_images=test_images/255

# print(train_images[0].shape)


# #building the model
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(100,100,3)),
#     keras.layers.Dense(128,activation = 'relu'), #activation if it passes certian threshold(relu: rectified linear unit)
#     keras.layers.Dense(2,activation='softmax')
#     ])

# #compile the model/properties of model
# model.compile(optimizer = 'adam',
#               loss = 'sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# #Train the model
# model.fit(train_images, train_labels, epochs=10) #accuracy can't be 100 because then its overfitting(memoriaing images), should be from 80 to 90
# #Training has ended

# #Now testing Begins
# #test the model
# print('test')
# test_loss, test_acc = model.evaluate(test_images, test_labels) #prediction on the test images
# print(test_acc)


# model.save('HumanOrNot.h5')


'''

image = cv2.imread('/Users/sasankgamini/Desktop/MachineLearningProjects/HumanIdentifier/humanIdentifier/testimagenothing.JPG')
resizedImage = cv2.resize(image, (100,100))
# cv2.imshow('nothing', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
reshapedImage = np.reshape(resizedImage, (1,100,100,3))
reshapedImage = reshapedImage/255

model = load_model('SmthnOrNothinImgClassifier.h5')
prediction = model.predict(reshapedImage)
print(prediction[0])
index = np.argmax(prediction[0])
print(index)
'''

capture = cv2.VideoCapture(0)
model = load_model('HumanOrNot.h5')
outcomes = ['something','nothing']

while True:
    _, frame = capture.read()
    resizedFrame = cv2.resize(frame, (100,100))
    reshapedFrame = np.reshape(resizedFrame, (1,100,100,3))
    reshapedFrame = reshapedFrame/255
    prediction = model.predict(reshapedFrame)
    index = np.argmax(prediction[0])
    cv2.putText(frame, outcomes[index],(100,100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,255),3)
    cv2.imshow('video', frame)
  
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()




