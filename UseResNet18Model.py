from tensorflow.python.keras.models import load_model

from ProcessImage import get_train_valid_test

x_train, x_valid, x_test, y_train, y_valid, y_test = get_train_valid_test()
model = load_model('img_classify.h5')
model.summary()
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"accuracy:{acc}\nloss:{loss}")