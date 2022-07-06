
import numpy as np 
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
#model loading
model=load_model('model.hdf5')#model filename
datagenerator=ImageDataGenerator(preprocessing_function=preprocess_input)
#code testing


test_generator=datagenerator.flow_from_directory('test',
    target_size=(200,150),
    batch_size=1,
    class_mode=None)#(boolean value)


#prediction
pred=model.predict_generator(test_generator,steps=len(test_generator),verbose=1)#load cheyythe kannan verbose=1 visual representation
print(pred)
#predicted_class_indices ithil 2 posibility ind negative kittanum positive kittanum nammal predict cheyythalil max value eduthitt athil max ethano positive or
#negative predict cheyyum (value index)
predicted_class_indices=np.argmax(pred,axis=1)
label=['negative','positive']
out=label[predicted_class_indices[0]]
print(out)

#daily update
file= open('out.txt','w')
file.write(out)
file.close()

from datetime import date

today = date.today()

# dd/mm/YY
curr_date = today.strftime("%d/%m/%Y")
print("Todays date:",curr_date)

file=open("date.txt","r")
t_date=file.read()
file.close()

#print(t_date)

file=open("daily_update.txt","r")
mal_count=int(file.read())
file.close()
if curr_date==t_date:
	if out=='positive':
		mal_count+=1
else:
	file=open("date.txt","w")
	file.write(curr_date)
	file.close()
	if out=='positive':
		mal_count=1
	else:
		mal_count=0
		
file=open("daily_update.txt","w")
file.write(str(mal_count))
file.close()

print("Today's malaria cases:",mal_count)