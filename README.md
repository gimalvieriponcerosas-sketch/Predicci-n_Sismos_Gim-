# SISMO AI PERU

Aplicacion Flask para estimar categorias didacticas de intensidad mediante KNN.

## Requisitos

- Python 3.10 o superior

## Ejecucion

```powershell
py -m pip install -r requirements.txt
py app.py
```

Abrir http://127.0.0.1:5000

## Verificacion

```powershell
py -m unittest -v
```

El proyecto incluye validacion de parametros, salida en formato JSON y una interfaz adaptable a dispositivos moviles.

Cada estimacion completada se guarda en `instance/predictions.sqlite3` y aparece en **Reportes → Historial de estimaciones**. El contador del dashboard y los gráficos de estimaciones se actualizan con esos registros. Se puede borrar una estimacion o vaciar todo el historial desde esa pantalla. El historial también está disponible en `GET /api/predictions` (muestra las 100 estimaciones más recientes).

## Nota

El KNN utiliza datos ficticios y no constituye una prediccion real ni una alerta oficial.
