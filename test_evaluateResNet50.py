from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas

test_model = "resnet50" # model name

num_classes = 3

# Load data
print("ResNet50 loading information")
from tensorflow.keras.applications.resnet import preprocess_input
data_directory = './test/'

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
test_it = test_datagen.flow_from_directory(data_directory, class_mode='categorical', target_size = (200, 200), batch_size = 32)

# Download pretrained model
print("ResNet50 load pretrained model")
from tensorflow.keras.applications import ResNet50
resnet_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (200, 200, 3))
print("Number layers of ResNet50", len(resnet_model.layers))

print("ResNet50 remove some layers")
for layer in resnet_model.layers:
    layer.trainable = False

# Added new layers
print("ResNet50 adding some layers")
x = resnet_model.output
x = Flatten()(x)
x = Dense(128, activation = 'relu')(x)
x = Dense(num_classes, activation = 'softmax')(x)
transfer_model = Model(inputs = resnet_model.input, outputs = x)

# Display model structure
#transfer_model.summary()

# Load weight
print("ResNet50 loading model name")
modelPath = './modelStorage/resnet/'
modelFolder = 'Dec-20-2021-22-37-41' + '/'
modelName = "weights-improvement-53-0.43.hdf5"
    
transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.000001), metrics=["accuracy", "Precision", "Recall", "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives"])
transfer_model.load_weights(modelPath + modelFolder + modelName)
print("Finishing loading model")


# Evaluate model
loss, acc, preci, rec, tp, tn, fp, fn = transfer_model.evaluate(test_it, verbose=1)
specificity = (tn / (tn + fp))
F1score = 2*(rec*preci)/(rec+preci)
print("Specificity: ", specificity)
print("F1-score: ", F1score)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

score = pandas.DataFrame([[loss, acc, preci, rec, specificity, F1score, tp, tn, fp, fn]], columns=['loss', 'accuracy', 'precision', 'recall','Specificity', 'F1score', 'true_positive', 'true_negative', 'false_positive', 'false_negative'])
score.to_csv("./evaluateResult/" + "keras-" + test_model + '-' + modelName.split('.')[0] + ".csv")

# Confusion matrix
print("-------------------------------")
pred_y = transfer_model.predict(test_it, verbose=1)
predict = np.asarray(pred_y)
predict_class = np.argmax(predict, axis=1)
predict_class = predict_class.tolist()
print("-------------------------------")
cf_matrix = confusion_matrix(test_it.classes, predict_class)
print(cf_matrix)
print('Classification Report')
target_names = ['1', '2', '3']
report = classification_report(test_it.classes, predict_class, target_names=target_names, output_dict=True)
print(report)

# Save result
save_name = "./evaluateResult/" + "sklearn-" + test_model + '-' + modelName.split('.')[0] + ".csv"
df = pandas.DataFrame(report).transpose()
df.to_csv(save_name)

