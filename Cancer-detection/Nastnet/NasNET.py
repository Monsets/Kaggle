from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.nasnet import NASNetLarge
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from ..data.preprocessing import load_data

DATA_PATH = 'trainset-data.pic'
LABELS_PATH = 'trainset-data.pic'

input_tensor = Input(shape=(32, 32, 3))

model = NASNetLarge(input_tensor=input_tensor, weights=None, include_top = False)

# add a global spatial average pooling layer
output = model.output
output = GlobalAveragePooling2D()(output)
# let's add a fully-connected layer
output = Dense(1024, activation='relu')(output)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(output)

# this is the model we will train
model = Model(inputs=model.input, outputs=predictions)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='Adam', loss='categorical_crossentropy')

x, y = load_data(DATA_PATH, LABELS_PATH)
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
#data augmentation  using flips
idg = ImageDataGenerator(horizontal_flip = True, vertical_flip = True)

model.fit_generator(idg.flow(x_train, y_train, batch_size=64), steps_per_epoch = len(x_train) / 64, epochs = 100,
                    validation_data = idg.flow(x_test, y_test, batch_size = 64), validation_steps = len(x_test) / 64,
                    callbacks = ModelCheckpoint('checkpoints/nasnet-cancer-{val_loss:.2f}.hdf5', save_best_only = True))