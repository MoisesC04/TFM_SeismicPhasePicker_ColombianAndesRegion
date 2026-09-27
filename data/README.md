# Datos / Data

Este README está disponible en español e inglés. Describe la organización de `data/` en la copia local del proyecto y distingue los archivos preparados para el repositorio de los que se conservan únicamente en local. No certifica una nueva ejecución del flujo.

This README is available in Spanish and English. It describes the organization of `data/` in the local project copy and distinguishes files prepared for the repository from files retained locally only. It does not certify a fresh run of the workflow.

## Español

Los datos siguen las etapas del trabajo: catálogos y señales originales, inventarios tras el control de calidad, ventanas pseudoetiquetadas, conjunto para el ajuste fino y resultados de evaluación. La ausencia de algunos archivos voluminosos en GitHub responde a límites prácticos de almacenamiento y transferencia, no a que formen parte de una etapa distinta del estudio.

| Ruta | Contenido |
| --- | --- |
| `raw/catalog/` | Catálogos de eventos y estaciones utilizados para solicitar los registros. |
| `raw/waveforms/` | Formas de onda originales descargadas; se conservan solo en local. |
| `interim/preprocessing/` | Inventarios de ventanas y resultados del control de calidad. |
| `processed/labeling/` | Tablas de ventanas con categoría asignada mediante consenso. |
| `processed/unlabeling/` | Tablas de ventanas sin categoría definitiva. |
| `processed/dataset/metadata.csv` | Metadatos y particiones del conjunto final. |
| `processed/dataset/waveforms.hdf5` | Señales del conjunto final; se conserva solo en local. |
| `processed/models/` | Pesos preentrenados y ajustados de las cuatro arquitecturas. |
| `processed/eval_results/` | Tablas, resúmenes y algunas figuras de evaluación. Los registros de inferencia grandes `records_*.pkl` se conservan solo en local. |
| `processed/images/` | Figuras de apoyo organizadas por tema. |
| `samples/waveforms/` | Pequeña muestra de señales SAC para inspeccionar el formato de los registros. |

Las formas de onda originales y `waveforms.hdf5` no se suben por su tamaño; tampoco se incluyen los registros intermedios de inferencia `records_*.pkl`. Por ello, clonar el repositorio permite consultar el código, los archivos ligeros, los pesos y los resultados tabulados disponibles, pero **no equivale a disponer de todos los datos necesarios para repetir el flujo completo**. Las señales originales pueden solicitarse mediante los servicios de la Red Sismológica Nacional de Colombia utilizados en la etapa de adquisición. Con las señales y el procesamiento descrito en `src/`, el archivo HDF5 y los registros intermedios pueden volver a generarse. Su regeneración no se ha comprobado en una instalación nueva de este repositorio.

Las categorías de `processed/labeling/` son pseudoetiquetas obtenidas por acuerdo entre modelos, no marcas manuales independientes. Los pesos de `processed/models/` representan entrenamientos ya realizados: no deben sobrescribirse al explorar los datos o el código.

## English

The data follow the stages of the study: original catalogs and signals, quality-control inventories, pseudo-labeled windows, the fine-tuning dataset, and evaluation results. Some large files are absent from GitHub because of practical storage and transfer constraints, not because they belong to a separate stage of the study.

| Path | Contents |
| --- | --- |
| `raw/catalog/` | Event and station catalogs used to request the records. |
| `raw/waveforms/` | Downloaded original waveforms; retained locally only. |
| `interim/preprocessing/` | Window inventories and quality-control results. |
| `processed/labeling/` | Tables of windows assigned a category by consensus. |
| `processed/unlabeling/` | Tables of windows without a definitive category. |
| `processed/dataset/metadata.csv` | Metadata and splits of the final dataset. |
| `processed/dataset/waveforms.hdf5` | Signals in the final dataset; retained locally only. |
| `processed/models/` | Pretrained and fine-tuned weights for the four architectures. |
| `processed/eval_results/` | Evaluation tables, summaries, and selected figures. Large inference records (`records_*.pkl`) are retained locally only. |
| `processed/images/` | Supporting figures organized by topic. |
| `samples/waveforms/` | Small sample of SAC signals for inspecting the record format. |

The original waveforms and `waveforms.hdf5` are not uploaded because of their size; intermediate inference records (`records_*.pkl`) are also excluded. Consequently, cloning the repository provides access to the code, smaller files, weights, and available tabulated results, but **does not provide every input required to rerun the full workflow**. Original signals can be requested through the Red Sismológica Nacional de Colombia services used in the acquisition stage. Given those signals and the processing described in `src/`, the HDF5 file and intermediate records can be regenerated. Regeneration has not been verified in a fresh installation of this repository.

The categories in `processed/labeling/` are pseudo-labels obtained through model agreement, not independent analyst picks. The weights in `processed/models/` are the result of completed training runs and should not be overwritten when exploring the data or code.
