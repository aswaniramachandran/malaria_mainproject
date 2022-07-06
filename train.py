


#model building use cheyyunna libraries
#libraries
import keras
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Sequential
from keras import optimizers
from keras import  regularizers
from keras.layers.normalization import BatchNormalization
 #model building 
#convolution layer adding
#filtersize(64)(kernalsize 4,4)
def cnnsvm():
    #sequential function(model variable vechu)
    model=Sequential()
    #layers adding
    model.add(Conv2D(64,kernel_size=(4,4),activation="relu",input_shape=(200,150,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,4)))
    #svm non linear regularizer function implementation using cnn
    model.add(Conv2D(64,kernel_size=(3,5),activation="relu",kernel_regularizer=regularizers.l2(0.04)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,kernel_size=(3,5),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    #input layer il ninu output ayitt ethra values vennam ennanu(hidden layer) using ANN (Neruron)
    model.add(Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation="relu",kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.1))
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(2,activation="softmax"))
    #loss function used  specify the datatype(catagorical anno continues anno)
    # metrix vechitt model evaluate cheyyunne
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.9,epsilon=1e-8,decay=0.0),metrics=["accuracy"])
    model.summary()
    return model
    
model=cnnsvm()
    
    
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

datagenerator=ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator=datagenerator.flow_from_directory('dataset/train',
    target_size=(200,150),
    batch_size=15,
    class_mode='categorical')
    
    
    
test_generator=datagenerator.flow_from_directory('dataset/test',
    target_size=(200,150),
    batch_size=2,
    class_mode='categorical')

#train cheyyan 
fit_history=model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=1,
        validation_data=test_generator,
        validation_steps=20
        )

    
    
    
model.save("model.hdf5")    
