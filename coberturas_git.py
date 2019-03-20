import numpy as np
import statsmodels.api as sm
import pandas_market_calendars as mcal
import datetime
from dataclass import Dataholder
import quandl
import pandas as pd
import matplotlib.pyplot as plt


#levantar base
futures = pd.read_pickle( "base.pkl")

my_tuple = pd.DataFrame(futures).shape
num_contracts = my_tuple[1]
num_days = my_tuple[0]



def calculartau(contrato, periodo = 0):
    """
    calcula el tiempo al vencimiento en años de contratos mensuales sucesivos
    contrato = 0 significa vencimiento a fin de mes
    contrato = 1 vencimiento a fin del mes siguente, etc
    """

    while periodo > 30:
        periodo = periodo - 30

    cant = contrato + 1
    tau = 30 * cant - periodo

    return tau

def cobertura(
    datos_futuros, datos_spot ,
    exposicion = 1000 , periodo_total = 5 ,
    fecha_inicio= "2006-02-03", fecha_fin = "2018-08-29",
    frecuencia_rebalanceo = 12 , cantidad_de_futuros = 1,
    tipo = "naive", modelo_medias = [], modelo_covarianzas = []):

    """
    Instrumenta una cobertura dados varios parámetros:
    datos: precios de futuros y spot por separado
    exposicion: puede ser positiva o negativa (en unidades a las que esta el precio spot, default mil unidades (barriles, toneladas, onzas, etc))
    periodo_total: periodo en el que vence la exposicion en años (default 5 años)
    frecuencia_rebalanceo: cada cuanto rebalanceo (en veces por años, default mensual)
    cantidad_de_futuros: cuantos contratos voy a usar para instrumentar la cobertura (default 1)
    tipo: que tipo de cobertura quiero instrumentar,
    tipos validos: naive(default), minima varianza (3 versiones), implicita
    modelo_medias: state_means del filtro de kalman del modelo a tener en cuenta (cuando aplique)
    modelo_covarianzas: state_covs del filtro de kalman del modelo a tener en cuenta (cuando aplique)
    """
    ratio_cobertura = []

    valor_portafolio = []

    posicion = []

    keys = []

    if all([fecha_inicio, fecha_fin]) is not None:

        calendario = mcal.get_calendar("ICEUS").schedule(fecha_inicio,fecha_fin)
        total_dias = datos_futuros.shape[0]




    if tipo == "naive":
        """
        Instrumenta una cobertura naive
        toma una posicion contraria a la exposicion
        con el contrato mas cercano al vencimiento
        para cada periodo de rebalanceo
        este ya esta
        """

        for periodo in range((total_dias) - 1 ):

            _ = np.zeros(cantidad_de_futuros)
            posi_ind = []
            keys.append(periodo)

            for contrato in range(cantidad_de_futuros):

                _[contrato] = datos_futuros.iloc[periodo,contrato]

                posi_ind.append(-exposicion)

            posicion.append(posi_ind)

            valor_portafolio.append(exposicion*(datos_spot[periodo]) + sum(np.array(posicion[periodo]*_)))


            del _ , posi_ind

        return dict(zip(keys,valor_portafolio )) , posicion




    elif tipo == "minima_varianza_SM":
        """
        cobertura que minimiza la varianza del portafolio cuebierto
        toma la posicion dada por un ratio de cobertura que es un cociente
        de varianzas y covarianzas que estan calculadas segun el modelo de S_M

        """

        total_dias = modelo_covarianzas.shape[0]
        cantidad_de_futuros = 2
        covar = np.empty((total_dias, cantidad_de_futuros))
        var = np.empty((total_dias))

        for periodo in range((total_dias) - 1 ):

            _ = np.zeros(cantidad_de_futuros)

            var[periodo] = modelo_covarianzas[periodo][0][0] + 2 * modelo_covarianzas[periodo][0][1] + modelo_covarianzas[periodo][1][1]


            for contrato in range(cantidad_de_futuros):

                covar[periodo][contrato] = 1/2 * modelo_covarianzas[periodo][0][1] + 1/2 * modelo_covarianzas[periodo][1][1]

                _[contrato] = - datos_futuros.iloc[periodo, contrato] * covar[periodo][contrato] / var[periodo]

            posicion.append(_)

            valor_portafolio.append(exposicion*(datos_spot[periodo]) + sum(np.array(exposicion * (_))))

            keys.append(periodo)

            del _

        return dict(zip(keys, valor_portafolio)), posicion


    elif tipo == "minima_varianza_CIR":
        """
        cobertura que minimiza la varianza del portafolio cuebierto
        toma la posicion dada por un ratio de cobertura que es un cociente
        de varianzas y covarianzas que estan calculadas segun el modelo de CIR
        """
        total_dias = modelo_covarianzas.shape[0]
        cantidad_de_futuros = 2
        covar = np.empty((total_dias, cantidad_de_futuros))
        var = np.empty((total_dias))

        for periodo in range((total_dias) - 1 ):

            _ = np.zeros(cantidad_de_futuros)

            var[periodo] = modelo_covarianzas[periodo][0][0] + 2 * modelo_covarianzas[periodo][0][1] + modelo_covarianzas[periodo][1][1]


            for contrato in range(cantidad_de_futuros):

                covar[periodo][contrato] = 1/2 * modelo_covarianzas[periodo][0][1] + 1/2 * modelo_covarianzas[periodo][1][1]

                _[contrato] = - datos_futuros.iloc[periodo, contrato] * covar[periodo][contrato] / var[periodo]

            posicion.append(_)

            valor_portafolio.append(exposicion*(datos_spot[periodo]) + sum(np.array(exposicion * (_))))

            keys.append(periodo)

            del _

        return dict(zip(keys, valor_portafolio)), posicion

    elif tipo == "minima_varianza_regresion":
        """
        cobertura que minimiza la varianza del portafolio cubierto
        toma la posicion dada por un ratio de cobertura calculado
        como el coeficiente de una regresion
        esta anda
        """
        import statsmodels.api as sm

        for periodo in range((total_dias) - 1 ):
            if periodo < 50:
                pass

            else:
                Y = datos_spot[:periodo]
                X = datos_futuros[:periodo]
                X = sm.add_constant(X)
                _ = np.zeros(cantidad_de_futuros)

                for contrato in range(cantidad_de_futuros):

                    modelo_reg = sm.OLS(Y,X).fit()

                    _[contrato] = - modelo_reg.params[contrato + 1] * datos_futuros.iloc[periodo, contrato]


                posicion.append(_)

                valor_portafolio.append(exposicion*(datos_spot[periodo]) + sum(np.array(exposicion * (_))))

                keys.append(periodo)

                del _, X, Y

        return dict(zip(keys, valor_portafolio)), posicion


    elif tipo == "minima_varianza_datos_historicos":
        """
        cobertura que minimiza la varianza del portafolio cuebierto
        toma la posicion dada por un ratio de cobertira que es un cociente
        de varianzas y covarianzas que estan calculadas con datos historicos
        """


        for periodo in range((total_dias) - 1 ):
            if periodo < 50:
                pass

            else:
                var = np.var(np.array(datos_spot[:periodo]))
                _ = np.zeros(cantidad_de_futuros)

                covar = np.cov(np.array(datos_spot[:periodo]) ,np.array(datos_futuros.iloc[:periodo,:]), rowvar = False)


                for contrato in range(cantidad_de_futuros):


                    _[contrato] = -datos_futuros.iloc[periodo,contrato] * covar[0,contrato + 1] / var


                posicion.append(_)

                valor_portafolio.append(exposicion*(datos_spot[periodo]) + sum(np.array(exposicion * np.array(_))))

                keys.append(periodo)

                del _, var, covar

        return dict(zip(keys, valor_portafolio)), posicion

    else:

        print("""El usuario debe introducir una metodologia de cobertura contemplada por el codigo.
        Metodologias contempladas:
        Cobertura naive (keyword: naive);
        Cobertura de minima varianza, modelo de Schwartz-Smith de dos factores (keyword: minima_varianza_SM);
        Cobertura de minima varianza, modelo con volatilidad heteroscedastica (keyword: minima_varianza_CIR)
        Cobertura de minima varianza, enfoque de regresion (keyword:minima_varianza_regresion)
        Cobertura de minima varianza, datos historicos (keyword:minima_varianza_regresion)
                """)


## Armado de coberturas

#Varianza Minima Regresion

b, posicion_b = cobertura(futures.iloc[:, 1::] ,futures.iloc[:,0], cantidad_de_futuros = num_contracts-1, tipo ="minima_varianza_regresion")

fig, ax = plt.subplots(figsize = (15,9))
plt.plot(b.keys(), b.values())

fig, ax = plt.subplots(figsize = (15,9))
plt.plot(b.keys(), posicion_b )

# Varianza Minima, datos historicos

c, posicion_c = cobertura(futures.iloc[:, 1::] ,futures.iloc[:,0], cantidad_de_futuros = num_contracts-1, tipo ="minima_varianza_datos_historicos")

fig, ax = plt.subplots(figsize = (15,9))
plt.plot(c.keys(), c.values())

fig, ax = plt.subplots(figsize = (15,9))
plt.plot(c.keys(), posicion_c )

#modelo CIR


state_CIR = np.load("cov_modeloCIR_crudo.npy")

state_CIR.shape

d, posicion_d = cobertura(futures.iloc[:, 1::] ,futures.iloc[:,0],
                        cantidad_de_futuros = num_contracts-1, tipo = "minima_varianza_CIR",
                        modelo_covarianzas = state_CIR)


fig, ax = plt.subplots(figsize = (15,9))
plt.plot(d.keys(), d.values())

fig, ax = plt.subplots(figsize = (15,9))
plt.plot(d.keys(), posicion_d )


# modelo SM

state_covs = np.load("cov_estados_crudo_SM.npy")

state_covs.shape[0]

e, posicion_e = cobertura(futures.iloc[:, 1::] ,futures.iloc[:,0],
                        cantidad_de_futuros = num_contracts-1, tipo = "minima_varianza_SM",
                        modelo_covarianzas = state_covs)


fig, ax = plt.subplots(figsize = (15,9))
plt.plot(e.keys(), e.values())

fig, ax = plt.subplots(figsize = (15,9))
plt.plot(e.keys(), posicion_e )



# modelo SM

state_covs = np.load("cov_estados_crudo_SM.npy")

state_covs.shape[0]

e, posicion_e = cobertura(futures.iloc[:, 1::] ,futures.iloc[:,0],
                        cantidad_de_futuros = num_contracts-1, tipo = "minima_varianza_SM",
                        modelo_covarianzas = state_covs)


fig, ax = plt.subplots(figsize = (15,9))
plt.plot(e.keys(), e.values())

fig, ax = plt.subplots(figsize = (15,9))
plt.plot(e.keys(), posicion_e )
