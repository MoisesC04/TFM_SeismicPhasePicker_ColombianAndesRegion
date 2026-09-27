# Código y orden de ejecución / Code and execution order

Este README está disponible primero en español y después en inglés. Describe la función prevista de los archivos según la lectura del código y las salidas conservadas en el proyecto; no certifica una nueva ejecución integral del flujo.

This README is provided first in Spanish and then in English. It describes the intended role of each file based on code inspection and the outputs retained in the project; it does not certify a fresh end-to-end run.

## Español

Este directorio reúne el flujo empleado para obtener registros de la RSNC, acondicionarlos, generar pseudoetiquetas por consenso, construir el conjunto de datos, ajustar cuatro arquitecturas y evaluar sus resultados. Las carpetas de `validation/` y `visualization/` contienen comprobaciones y figuras de apoyo; no son pasos obligatorios del flujo principal.

### Antes de comenzar

- Las rutas de varios archivos apuntan a la ubicación local del proyecto. Si se usa otra carpeta, hay que revisar esas rutas antes de ejecutar los notebooks.
- Los catálogos de entrada están en `data/raw/catalog/`. Las formas de onda originales de `data/raw/waveforms/` y el archivo `data/processed/dataset/waveforms.hdf5` no se distribuyen en GitHub por su tamaño.
- Los pesos ya entrenados están en `data/processed/models/`. Volver a ejecutar los notebooks de ajuste fino puede sobrescribirlos; para consultar los resultados existentes no es necesario repetir el entrenamiento.
- Las métricas de evaluación comparan las predicciones con pseudoetiquetas de consenso, no con marcas manuales independientes.

### Flujo principal

Los pasos 1 a 4 son secuenciales. Después, cada arquitectura puede tratarse por separado, pero su notebook `collect_*.ipynb` necesita los pesos generados por el correspondiente `finetune_*.ipynb`. Las dos evaluaciones se realizan una vez disponibles los registros de inferencia de las arquitecturas que se desean comparar.

1. **Adquisición.** `acquisition/request_SGC_seismograms.ipynb` consulta el servicio de la RSNC a partir de los catálogos de eventos y estaciones, y guarda las señales descargadas en `data/raw/waveforms/`.
2. **Control de calidad y acondicionamiento.** `preprocessing/qc_preprocessing.ipynb` revisa las componentes, acondiciona las ventanas y genera el inventario de registros válidos en `data/interim/preprocessing/`.
3. **Etiquetado por consenso.** `labeling/consensus_labeling.ipynb` aplica las cuatro arquitecturas preentrenadas, comprueba el acuerdo entre sus propuestas de fases y guarda las ventanas etiquetadas y no etiquetadas en `data/processed/labeling/` y `data/processed/unlabeling/`.
4. **Construcción del conjunto.** `dataset/build_dataset.ipynb` reúne las ventanas seleccionadas, incorpora ruido real y sintético, y prepara las particiones para el ajuste fino. Produce `data/processed/dataset/metadata.csv` y `waveforms.hdf5` en esa misma carpeta.
5. **Ajuste fino y recopilación de predicciones.** En cada fila se ejecuta primero el ajuste y luego la recopilación. Los notebooks `collect_*` guardan las predicciones de las versiones preentrenadas y ajustadas para su evaluación posterior.

   | Arquitectura | Ajuste fino | Recopilación |
   | --- | --- | --- |
   | PhaseNet | `training/phasenet/finetune_phasenet.ipynb`: ajusta el selector y guarda sus pesos. | `training/phasenet/collect_phasenet.ipynb`: obtiene y almacena sus predicciones. |
   | EQTransformer | `training/eqtransformer/finetune_eqtransformer.ipynb`: ajusta el selector y guarda sus pesos. | `training/eqtransformer/collect_eqtransformer.ipynb`: obtiene y almacena sus predicciones. |
   | EQCCT | `training/eqcct/finetune_eqcct.ipynb`: ajusta los componentes de P y S y guarda sus pesos. | `training/eqcct/collect_eqcct.ipynb`: obtiene y almacena sus predicciones. |
   | GPD | `training/gpd/finetune_gpd.ipynb`: ajusta el selector y guarda sus pesos. | `training/gpd/collect_gpd.ipynb`: obtiene y almacena sus predicciones. |

6. **Evaluación.** `evaluation/evaluate_p1_picking.ipynb` calcula las métricas de detección y precisión temporal, incluida la comparación sobre el conjunto común de la onda S. Después, `evaluation/evaluate_p2_analysis.ipynb` examina curvas precisión-recuperación, ruido real y variaciones según las condiciones de registro. Sus tablas y figuras se guardan principalmente en `data/processed/eval_results/` y `data/processed/images/evaluation/`.

### Módulos de apoyo

Estos archivos se importan desde los notebooks correspondientes; no constituyen pasos independientes que deban ejecutarse en orden:

- `preprocessing/preprocessing_utils.py`: lectura de señales y funciones de acondicionamiento y control de calidad.
- `labeling/consensus_utils.py`: carga de selectores preentrenados y funciones para decidir el consenso de fases.
- `training/training_utils.py`: carga del conjunto, entrenamiento, inferencia y almacenamiento de métricas y registros.
- `evaluation/evaluation_utils.py`: funciones compartidas para métricas, análisis y figuras de evaluación.

### Comprobaciones y figuras opcionales

Estos archivos ayudan a inspeccionar etapas concretas. No es necesario ejecutarlos para seguir el flujo principal:

- `labeling/validation/consensus_gpd_aic_test.ipynb`: prueba diagnóstica histórica del consenso, GPD y AIC; no forma parte del etiquetado principal.
- `dataset/validation/verify_synth_noise.ipynb`: compara el ruido sintético con las señales de partida.
- `dataset/validation/verify_synth_noise_clean.ipynb`: revisa ejemplos de ruido sintético alejados de la llegada P.
- `dataset/validation/verify_critical_events.ipynb`: inspecciona eventos en los que la extracción de ruido puede ser crítica.
- `evaluation/weak_phase_snr.py`: análisis complementario de fases débiles y relación señal-ruido a partir de registros de evaluación.
- `visualization/data_preprocessing_figures.ipynb`: figuras de registros antes y después del acondicionamiento.
- `visualization/figura_snr.ipynb`: figura de los intervalos usados para calcular la relación señal-ruido en un ejemplo.
- `visualization/figura_snr_marco_teorico.ipynb`: ilustración de esos intervalos para el marco teórico.
- `visualization/figure_synthetic_noise_ZAR_2018_344.ipynb`: muestra un ejemplo de señal y ruido sintético; el notebook no guarda la figura automáticamente.
- `visualization/labeling_results_report.ipynb`: resume las tablas de etiquetado y exporta CSV a `data/processed/eval_results/labeling_report/`.
- `visualization/figura_objetivo_perdidas.py`: genera `curvas.tex`, un recurso gráfico auxiliar, en el directorio desde el que se ejecute.

### Entradas y salidas principales

| Ruta desde la raíz del proyecto | Contenido |
| --- | --- |
| `data/raw/catalog/` | Catálogos de eventos y estaciones usados en la adquisición. |
| `data/raw/waveforms/` | Formas de onda originales; se conservan localmente y no se suben a GitHub. |
| `data/interim/preprocessing/` | Inventarios de ventanas y resultados del control de calidad. |
| `data/processed/labeling/` y `data/processed/unlabeling/` | Ventanas con categoría por consenso y ventanas sin categoría definitiva. |
| `data/processed/dataset/metadata.csv` | Metadatos y particiones del conjunto final. |
| `data/processed/dataset/waveforms.hdf5` | Señales del conjunto final; se conserva localmente y no se sube a GitHub. |
| `data/processed/models/` | Pesos preentrenados y ajustados; conservar sin sobrescribir. |
| `data/processed/eval_results/` | Registros de inferencia, métricas y tablas de evaluación. Los archivos grandes `records_*.pkl` se conservan solo localmente. |
| `data/processed/images/` | Figuras de evaluación y material gráfico generado para el proyecto. |

## English

This directory contains the workflow for acquiring RSNC records, preprocessing them, generating consensus-based pseudo-labels, building the dataset, fine-tuning four architectures, and evaluating their results. The `validation/` and `visualization/` directories contain supporting checks and figures; they are not required stages of the main workflow.

### Before you begin

- Several files use paths tied to the local project location. Review these paths before running the notebooks from a different directory.
- Input catalogs are in `data/raw/catalog/`. The original waveforms in `data/raw/waveforms/` and `data/processed/dataset/waveforms.hdf5` are not distributed through GitHub because of their size.
- Trained weights are already stored in `data/processed/models/`. Rerunning the fine-tuning notebooks may overwrite them; reviewing existing results does not require retraining.
- Evaluation metrics compare predictions with consensus pseudo-labels, not with independent analyst picks.

### Main workflow

Steps 1–4 are sequential. Each architecture can then be handled separately, but its `collect_*.ipynb` notebook requires the weights produced by the corresponding `finetune_*.ipynb`. Run the two evaluation stages once inference records are available for the architectures being compared.

1. **Acquisition.** `acquisition/request_SGC_seismograms.ipynb` queries the RSNC service using the event and station catalogs and saves downloaded signals under `data/raw/waveforms/`.
2. **Quality control and preprocessing.** `preprocessing/qc_preprocessing.ipynb` checks the components, preprocesses the windows, and creates an inventory of valid records in `data/interim/preprocessing/`.
3. **Consensus labeling.** `labeling/consensus_labeling.ipynb` applies the four pretrained architectures, checks agreement among their phase picks, and stores labeled and unlabeled windows in `data/processed/labeling/` and `data/processed/unlabeling/`.
4. **Dataset construction.** `dataset/build_dataset.ipynb` combines the selected windows, incorporates real and synthetic noise, and prepares the fine-tuning splits. It is designed to produce `data/processed/dataset/metadata.csv` and `waveforms.hdf5` in the same directory.
5. **Fine-tuning and prediction collection.** Within each row, fine-tuning precedes collection. The `collect_*` notebooks store predictions from pretrained and fine-tuned versions for subsequent evaluation.

   | Architecture | Fine-tuning | Collection |
   | --- | --- | --- |
   | PhaseNet | `training/phasenet/finetune_phasenet.ipynb`: fine-tunes the picker and saves its weights. | `training/phasenet/collect_phasenet.ipynb`: obtains and stores its predictions. |
   | EQTransformer | `training/eqtransformer/finetune_eqtransformer.ipynb`: fine-tunes the picker and saves its weights. | `training/eqtransformer/collect_eqtransformer.ipynb`: obtains and stores its predictions. |
   | EQCCT | `training/eqcct/finetune_eqcct.ipynb`: fine-tunes its P- and S-phase components and saves their weights. | `training/eqcct/collect_eqcct.ipynb`: obtains and stores its predictions. |
   | GPD | `training/gpd/finetune_gpd.ipynb`: fine-tunes the picker and saves its weights. | `training/gpd/collect_gpd.ipynb`: obtains and stores its predictions. |

6. **Evaluation.** `evaluation/evaluate_p1_picking.ipynb` computes detection and timing metrics, including the comparison on the common S-phase subset. `evaluation/evaluate_p2_analysis.ipynb` then examines precision–recall curves, real noise, and variation across recording conditions. The notebooks are designed to save tables and figures mainly under `data/processed/eval_results/` and `data/processed/images/evaluation/`.

### Support modules

These files are imported by the relevant notebooks; they are not independent stages to run in sequence:

- `preprocessing/preprocessing_utils.py`: signal reading, preprocessing, and quality-control functions.
- `labeling/consensus_utils.py`: pretrained picker loading and phase-consensus functions.
- `training/training_utils.py`: dataset loading, training, inference, and storage of metrics and records.
- `evaluation/evaluation_utils.py`: shared functions for evaluation metrics, analyses, and figures.

### Optional checks and figures

These files support inspection of specific stages. They are not required to follow the main workflow:

- `labeling/validation/consensus_gpd_aic_test.ipynb`: historical diagnostic test of consensus, GPD, and AIC; not part of the main labeling stage.
- `dataset/validation/verify_synth_noise.ipynb`: compares synthetic noise with the source signals.
- `dataset/validation/verify_synth_noise_clean.ipynb`: checks synthetic-noise examples away from the P arrival.
- `dataset/validation/verify_critical_events.ipynb`: inspects events for which noise extraction may be problematic.
- `evaluation/weak_phase_snr.py`: supplementary weak-phase and signal-to-noise analysis using evaluation records.
- `visualization/data_preprocessing_figures.ipynb`: figures of records before and after preprocessing.
- `visualization/figura_snr.ipynb`: illustrates the intervals used to calculate the signal-to-noise ratio in one example.
- `visualization/figura_snr_marco_teorico.ipynb`: illustrates those intervals for the theoretical framework.
- `visualization/figure_synthetic_noise_ZAR_2018_344.ipynb`: displays an example signal and synthetic noise; it does not automatically save the figure.
- `visualization/labeling_results_report.ipynb`: summarizes labeling tables and exports CSV files to `data/processed/eval_results/labeling_report/`.
- `visualization/figura_objetivo_perdidas.py`: generates `curvas.tex`, an auxiliary graphic resource, in the directory from which it is run.

### Main inputs and outputs

| Path from the project root | Contents |
| --- | --- |
| `data/raw/catalog/` | Event and station catalogs used for acquisition. |
| `data/raw/waveforms/` | Original waveforms; retained locally and not uploaded to GitHub. |
| `data/interim/preprocessing/` | Window inventories and quality-control results. |
| `data/processed/labeling/` and `data/processed/unlabeling/` | Windows assigned a category by consensus and windows without a definitive category. |
| `data/processed/dataset/metadata.csv` | Metadata and splits of the final dataset. |
| `data/processed/dataset/waveforms.hdf5` | Signals in the final dataset; retained locally and not uploaded to GitHub. |
| `data/processed/models/` | Pretrained and fine-tuned weights; do not overwrite. |
| `data/processed/eval_results/` | Inference records, metrics, and evaluation tables. Large `records_*.pkl` files are retained locally only. |
| `data/processed/images/` | Evaluation figures and supporting graphics generated for the project. |
