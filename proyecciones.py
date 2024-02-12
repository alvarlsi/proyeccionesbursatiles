!pip install PyPortfolioOpt
!pip install yfinance --upgrade --no-cache-dir
!pip install pandas-datareader
!pip install yahoo_fin #--update
import yahoo_fin
from yahoo_fin.stock_info import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import seaborn as sns
import scipy.stats as stats

from datetime import datetime
import yfinance as yf

yf.pdr_override()

plt.style.use('seaborn-colorblind')

#####################################
##### lista de activos a analizar
lista=['AAPL','GOOGL','META','AMZN','MSFT']

#####################################
####  gráfico de valuaciones de activos

tickers =lista[:]

prices = pdr.get_data_yahoo(tickers, start = '2002-01-01', end ='2021-12-08', threads=False)['Adj Close']# dt.date.today())['Adj Close']

prices = pdr.get_data_yahoo(tickers, start = '2002-01-01', end ='2021-12-08', threads=False)['Adj Close']# dt.date.today())['Adj Close']

returns = prices.pct_change()
returns

plt.plot(prices)
plt.legend(prices.columns)
plt.show()

#####################################
#####  gráfico de estimaciones de un activo o de una inversión

fig, ax = plt.subplots(figsize=(12,8))

def mov_average(prices,ds_fut):
  promedio=prices.rolling(30).mean()
  return promedio

def grafico(ax,precios_fechas):
  if len(precios_fechas.columns)>1:
    plt.plot(precios_fechas/precios_fechas[:1].values)
  else:
    plt.plot(precios_fechas)
  plt.legend(precios_fechas.columns)
  return ax,precios_fechas

def promedio(precios_fechas,ds_fut):
  prom_activos=precios_fechas.mean(axis=1)                                        ## promedio de cada día , sumando todos los activos
  pronostico=mov_average(prom_activos,ds_fut)                                     #saca el promedio de lo últimos 30 días que definí en la función
  return pronostico

def pronostico(ax,precios_fechas,lista_puntos, ds_fut):                           ##viene toda la tabla de activos elegidos
  if len(precios_fechas.columns)>1:                                               ### chequeo si hay más de un activo
    prices_ppal=precios_fechas[tickers[0]]/precios_fechas[tickers[0]][:1].values
    precios_fechas=precios_fechas/precios_fechas[:1].values
  pronostico=promedio(precios_fechas,ds_fut)                                      ### mov_average(prices_ppal,ds_fut)
  plt.plot(pronostico.shift(ds_fut),linestyle="--", label='Estimación')           ### desplacé el promedio hacia adelante para que sea una estimación
  plt.legend(precios_fechas.columns)

  for punto in lista_puntos:                                                      ###con un línea muestro en qué día estimo, y cuántos días hacia adelante
    plt.scatter(prices_ppal.index[punto:punto+1],prices_ppal[punto:punto+1])
    punto_fut=punto+ ds_fut
    plt.scatter(pronostico.index[punto_fut:punto_fut+1],pronostico[punto:punto+1])

    plt.plot([prices_ppal.index[punto:punto+1],prices_ppal.index[punto_fut:punto_fut+1]],
             [prices_ppal[punto:punto+1]      ,pronostico[punto:punto+1]]   ,color='red')

  return ax

precios_fechas=prices[max(prices[tickers].idxmin()):'2016-02-17']

grafico(ax,precios_fechas)

#Puedo pronosticar las cotizaciones de un activo o la capitalización de una inversión.
#Si estimo la valuación de un activo dependiendo del comportamiento pasado de una cartera de activos (incluyendo su propias valuacioines pasadas), 
#voy a poner a este activo como primero en la lista de "tickers"
#Si solo quiero estimar la capitalización de una cartera de activos, no me preocupo por el orden de los activos en la lista "tickers"

fechas_referencia=[i for i in range(120,len(precios_fechas)-120,120)]
pronostico(ax,precios_fechas,fechas_referencia,40)  # entra con precios reales, y se normaliza dentro de la función solo para graficar
plt.show()

#####################################
####  análisis de pendiente móvil de un activo basado en los últimos n cierres 

def sacar_pendiente(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return slope

def pendiente(tickers):
  pendiente_conj=pd.DataFrame()

  for activo in tickers:
    pend=pd.DataFrame(precios_fechas[activo].rolling(window=40, min_periods=40).apply(sacar_pendiente, raw=False))
    pend.index.name='Fecha'
    try:
      pendiente_conj=pd.merge(pendiente_conj,pend, on='Fecha', how='outer')
    except:
      pendiente_conj=pend
  return pendiente_conj

pendiente_conj=pendiente(tickers)
pendiente_conj

########

def sacar_desv(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return std_err

def desv_std(tickers):
  desv_conj=pd.DataFrame()

  for activo in tickers:
    desv=pd.DataFrame(precios_fechas[activo].rolling(window=40, min_periods=40).apply(sacar_desv, raw=False))
    desv.index.name='Fecha'
    try:
      desv_conj=pd.merge(desv_conj,desv, on='Fecha', how='outer')
    except:
      desv_conj=desv
  return desv_conj

desv_conj=desv_std(tickers)
desv_conj

#####################################
### volatilidad anualizada basado en los últimos n cierres
def volatilidad(std):
  volat=np.sqrt(252)*std
  return volat

volatilidad(std_conj)