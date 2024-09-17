# Databricks notebook source
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# Definindo o caminho do arquivo parquet particionado no DBFS
output_path = "dbfs:/FileStore/shared_uploads/matsudrivernf@gmail.com/processed_data/*"

# COMMAND ----------

# Lendo os dados em formato Parquet
df = spark.read.format("parquet").load(output_path)

# COMMAND ----------

from pyspark.sql.types import DoubleType

numeric_cols = ['VALORPARCELA', 'SALDO_DEVEDOR', 'VALOR_BRUTO', 'TROCO']

for col_name in numeric_cols:
    df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

# COMMAND ----------

# As colunas categóricas precisam ser convertidas em valores numéricos. Usaremos o StringIndexer para isso.
from pyspark.ml import Pipeline

categorical_cols = ['NME_EMPRESA', 'NME_ORGAO', 'NME_DESCONTO', 'REGIME_JURIDICO', 'SITUACAO_SERVIDOR']

indexers = [StringIndexer(inputCol=column, outputCol=column+"_INDEXED") for column in categorical_cols]


# COMMAND ----------

# Selecionaremos algumas colunas para serem nossas features.
feature_cols = [
    'PARCELAS_PAGAS', 'IDADE', 'PRAZO', 'VALORPARCELA', 'SALDO_DEVEDOR',
    'VALOR_BRUTO', 'TROCO'
] + [column + "_INDEXED" for column in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# COMMAND ----------

# Definir a coluna label
from pyspark.sql.functions import when

df = df.withColumn("label", when(col("PARCELAS_INADIM") > 0, 1).otherwise(0))


# COMMAND ----------

# Criar o Pipeline e preparar os dados
pipeline = Pipeline(stages=indexers + [assembler])

# Ajustar o pipeline aos dados
pipelineModel = pipeline.fit(df)
df_transformed = pipelineModel.transform(df)

# Selecionar apenas as colunas necessárias
final_data = df_transformed.select("features", "label")


# COMMAND ----------

#  Dividir os dados em treino e teste
train_data, test_data = final_data.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

# Aumentar o número de maxBins
rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=10, maxBins=150)

# Treinar o modelo
rf_model = rf.fit(train_data)

# Fazer previsões nos dados de teste
predictions = rf_model.transform(test_data)

# Avaliar o modelo
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# COMMAND ----------

# Exibir as previsões
predictions.select("features", "label", "prediction").show()

# COMMAND ----------

# Fazer previsões nos dados de teste
predictions = rf_model.transform(test_data)

# COMMAND ----------

# Selecionar as labels verdadeiras e as predições
y_test = predictions.select("label").rdd.flatMap(lambda x: x).collect()
y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Gerar os gráficos com base nas predições

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report
class_report = classification_report(y_test, y_pred, output_dict=True)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# 1. Confusion Matrix
plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Inadimplente', 'Inadimplente'], rotation=45)
plt.yticks(tick_marks, ['Not Inadimplente', 'Inadimplente'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Adicionar valores a cada célula na matriz de confusão
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()

# 2. Classification Report (Precision, Recall, F1-score)
plt.figure(figsize=(10, 5))
plt.bar(range(len(class_report['1'])), list(class_report['1'].values()), align='center', color='orange')
plt.xticks(range(len(class_report['1'])), list(class_report['1'].keys()))
plt.title("Classification Report for Class 'Inadimplente'")
plt.ylabel('Score')
plt.ylim([0, 1])

# 3. ROC Curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Mostrar todos os gráficos
plt.show()

