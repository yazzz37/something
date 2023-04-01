import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import utils
from keras.models import Model
from tensorflow.keras.models import load_model
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
df = pd.read_csv("дота2.csv")
X = pd.DataFrame()
X = df[["Маг", "Ближний бой", "Рыба", "Мужской пол", "Наличие зубов", "Имеет хук", "Огнестрел", "Керри"]]
Y = pd.DataFrame()
Y = df[["Пудж"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size = 0.30, random_state=42)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
model = Sequential()
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='adamax', loss='binary_crossentropy',  metrics=['accuracy'])
his = model.fit(X_train, Y_train, batch_size=1, epochs=230, validation_split=0.2, verbose=1)

from tkinter import *
root = Tk()
root.geometry("500x500")
root.title("ПУДЖЖЖЖ")
mage = Entry(root)
fists = Entry(root)
fish = Entry(root)
man = Entry(root)
teeth = Entry(root)
hook = Entry(root)
weapon = Entry(root)
kerry = Entry(root)
manual = Label(root, text="(пишете 1 если да, иначе - нет)")
manual.pack()
l1 = Label(root, text="Это маг?")
l2 = Label(root, text="Это ближник?")
l3 = Label(root, text="Это рыба?")
l4 = Label(root, text="Это мужик?")
l5 = Label(root, text="У него есть зубы?")
l6 = Label(root, text = "Пудж, Пудж, Пуджжжж")
lhook = Label(root, text = "Есть ли хук?")
lweapon = Label(root, text = "Есть ли огнестрел?")
lkerry = Label(root, text = "Это керри?")
l1.pack()
mage.pack()
l2.pack()
fists.pack()
l3.pack()
fish.pack()
l4.pack()
man.pack()
l5.pack()
teeth.pack()
lhook.pack()
hook.pack()
lweapon.pack()
weapon.pack()
lkerry.pack()
kerry.pack()
l6.pack()
def res():
    reses = model.predict(np.array([[int(mage.get()), int(fists.get()), int(fish.get()), int(man.get()), int(teeth.get()), int(hook.get()), int(weapon.get()), int(kerry.get())],]))
    l6.config(text="Это Пудж с вероятностью " + str(reses))
    root.mainloop()
b = Button(root, text="Вычислить", command=res)
b.pack()

plt.figure(figsize=(18, 4))
plt.title('Доля ошибки на обучающих и провечных данных')
plt.plot(his.history['loss'][:], 'go', linewidth=1, markersize=4, linestyle='--',
         label='Доля ошибки на обучающих данных')
plt.plot(his.history['val_loss'][:], 'r', linewidth=1.5, markersize=1, linestyle='-',
         label='Доля ошибки на проверочных данных')

plt.xlabel('Эпоха обучения')
plt.ylabel('Доля ошибки'),
plt.legend()
plt.show()

