# Adaptación de selectores de fases sísmicas a los Andes colombianos

**Moisés Carvajal Angarita**<br>
Tutor: Alberto Partida Rodríguez<br>
Septiembre de 2026

**Trabajo de Fin de Máster**<br>
Máster Universitario en Análisis de Datos Masivos<br>
Escuela de Arquitectura, Ingeniería y Diseño<br>
Universidad Europea de Madrid

Este README está disponible en [español](#español) e [inglés](#english).

## Español

### Descripción del trabajo

Este repositorio reúne los componentes computacionales desarrollados en el Trabajo de Fin de Máster para adaptar selectores de fases sísmicas al dominio local de la Red Sismológica Nacional de Colombia. Ante la ausencia de marcas temporales de analistas para las llegadas de las fases P y S en los catálogos públicos empleados, el trabajo integra la generación de datos pseudoetiquetados con el ajuste fino de modelos de aprendizaje profundo.

Esta estrategia se organiza en dos bloques complementarios que conectan la obtención de datos locales con la evaluación de los modelos. El primero construye un conjunto de pseudoetiquetas mediante consenso entre **PhaseNet, EQTransformer, EQCCT y GPD** a partir de registros previamente acondicionados, mientras que el segundo utiliza ese material para ajustar las cuatro arquitecturas y comparar sus versiones preentrenadas y ajustadas bajo un protocolo común.

### Principales resultados

El conjunto final reúne **106 580 ventanas**, compuestas por registros sísmicos pseudoetiquetados, ruido real y ruido sintético. El ajuste fino mejoró principalmente la precisión de la onda P y la sensibilidad de la onda S, además de reducir las falsas alarmas sobre ruido real en tres arquitecturas. La precisión temporal no presentó una mejora uniforme y los resultados conservaron perfiles de desempeño diferenciados entre modelos.

El procedimiento establece un flujo trazable para generar datos locales y evaluar el ajuste fino. Las métricas expresan coincidencia con las pseudoetiquetas de consenso, mientras que el alcance de los resultados corresponde a las condiciones de registro representadas en el conjunto.

### Organización del repositorio

| Directorio | Contenido y documentación |
| --- | --- |
| [`src/`](src/README.md) | Notebooks y módulos para adquisición, acondicionamiento, consenso, construcción del conjunto, ajuste fino, evaluación y figuras de apoyo. Su README describe el orden de ejecución y los archivos auxiliares. |
| [`data/`](data/README.md) | Catálogos, inventarios, tablas de pseudoetiquetas, metadatos, pesos de los modelos, resultados de evaluación y figuras. Su README detalla las entradas y salidas del flujo. |
| `deliverables/` | Contiene el PDF final del Trabajo de Fin de Máster, `TFM_Depósito_Final_MoisésCarvajalAngarita.pdf`. |

### Disponibilidad de los datos y contacto

Las formas de onda originales, el conjunto de señales en formato HDF5 y los registros intermedios voluminosos de inferencia no se distribuyen en GitHub por su tamaño. El repositorio incluye el código, los metadatos, los pesos y los resultados disponibles para su consulta.

Para solicitar acceso a los datos no incluidos, puede enviarse una solicitud a **[mcarvajala04@gmail.com](mailto:mcarvajala04@gmail.com)**, indicando los archivos de interés y el propósito de su uso. El procedimiento de adquisición de las señales a través de los servicios de la RSNC se encuentra en [`src/acquisition/`](src/acquisition/).

## English

### Project overview

This repository contains the computational components of the master's thesis on adapting seismic phase pickers to the local domain of the Red Sismológica Nacional de Colombia. The public catalogs used in the project contain located seismic events but do not include analyst-provided arrival-time picks for P and S phases. The work addresses this limitation through pseudo-labeled data generation and deep learning model fine-tuning.

The research combines two complementary stages. The first generates a local pseudo-labeled dataset through consensus among **PhaseNet, EQTransformer, EQCCT, and GPD**, using preprocessed records. The second uses these data to fine-tune all four architectures and compare their pretrained and fine-tuned versions under a common protocol.

### Main findings

The final dataset comprises **106,580 windows**, including pseudo-labeled seismic records, real noise, and synthetic noise. Fine-tuning mainly improved P-wave precision and S-wave sensitivity, while reducing false alarms in real noise for three architectures. Timing precision did not improve uniformly, and differentiated model performance profiles remained after fine-tuning.

The procedure establishes a traceable workflow for generating local data and evaluating fine-tuning. The metrics measure agreement with consensus pseudo-labels, and the scope of the findings corresponds to the recording conditions represented in the dataset.

### Repository organization

| Directory | Contents and documentation |
| --- | --- |
| [`src/`](src/README.md) | Notebooks and modules for acquisition, preprocessing, consensus labeling, dataset construction, fine-tuning, evaluation, and supporting figures. Its README describes the execution order and auxiliary files. |
| [`data/`](data/README.md) | Catalogs, inventories, pseudo-label tables, metadata, model weights, evaluation results, and figures. Its README details workflow inputs and outputs. |
| `deliverables/` | Contains the final master's thesis PDF, `TFM_Depósito_Final_MoisésCarvajalAngarita.pdf`. |

### Data availability and contact

Original waveforms, the HDF5 waveform dataset, and large intermediate inference records are not distributed through GitHub because of their size. The repository includes the code, metadata, weights, and available results for consultation.

To request access to data not included in the repository, email **[mcarvajala04@gmail.com](mailto:mcarvajala04@gmail.com)**, specifying the files of interest and the intended use. The signal acquisition procedure through RSNC services is provided in [`src/acquisition/`](src/acquisition/).
