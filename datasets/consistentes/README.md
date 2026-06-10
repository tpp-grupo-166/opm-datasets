# Datasets con PVT consistente con la simulación

Estas son las versiones de los datasets cuyas columnas PVT (`Bo_rb_stb`, `Bg_rb_scf`,
`Rs_scf_stb`) son las que salieron de la simulación, consistentes con las presiones.

## Por qué existe esta carpeta

En `datasets/` (y en el repo original `ricomateo/opm-proof-of-concept`) las tablas PVT de
Volve y Norne se editaron después de correr las simulaciones: presión, producción e
inyección son idénticas a estas copias, pero Bo/Bg/Rs difieren (Rs hasta +135% en Norne)
y se agregó la columna `Presion_Inicial_Reservorio_psi`.

Para modelos puramente data-driven (Ridge, XGBoost, LSTM) esto no cambia nada, porque
descartan esas columnas por leakage o las usan de forma marginal. Pero cualquier modelo
de balance de materiales / PINN que use la PVT se rompe con la versión editada: el
baseline de tanque en Norne pasa de R² +0.65 (estos datos) a −7.99 (los editados).

Los notebooks de PINN (notebook 7) y los resultados del informe se calcularon con ESTAS
versiones. Si un script necesita la presión inicial, se deriva como la primera
`Presion_Reservorio_psi` de cada simulación.

## Archivos

| Archivo | md5 | Nota |
|---|---|---|
| `dataset_volve.csv` | `39d79dc90967a3ea8d3c5cee02467271` | difiere de `datasets/` |
| `dataset_norne.csv` | `4f3cae378de762055dd7a1f9eaff5313` | difiere de `datasets/` |
| `dataset_spe9.csv` | `2f835c9e5c93fd065b75b46af8ab5340` | idéntico a `datasets/` (se incluye para tener el set completo en un solo lugar) |
