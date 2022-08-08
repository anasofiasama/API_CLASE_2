import pickle
import pandas as pd
import numpy as np

country='Other'
variety='Other'
aroma=7.42
aftertaste= 7.32
acidity=7.42
body=7.25
balance= 7.33
moisture=0.0

#Como es un pipline no se pude ingresar directamente, si fuera una lista si. En este caso se hace un pandas.

cols= ['country_of_origin','variety','aroma','aftertaste','acidity','body','balance','moisture'] # cabecera de pandas
data=[country,variety,aroma,aftertaste,acidity,body,balance,moisture] # datos

posted=pd.DataFrame(np.array(data).reshape(1,8),columns=cols) # se creo el archivo
# falta abrir el archivo pickle
loaded_model=pickle.load(open('../models/coffee_model.pkl','rb'))
result=loaded_model.predict(posted)
text_result=result.tolist()[0]
print(text_result)