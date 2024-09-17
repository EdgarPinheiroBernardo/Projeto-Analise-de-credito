# Databricks notebook source
# importando as libs necessarias para o processamento dos dados
import re
from pyspark.sql.functions import col, regexp_replace

# COMMAND ----------

# Definindo o caminho do arquivo no DBFS
file_path = "dbfs:/FileStore/shared_uploads/matsudrivernf@gmail.com/Siape_Saldo.csv"

# COMMAND ----------

# Lendo o arquivo CSV usando o PySpark
df = ( 
      spark.read.format("csv") 
        .option("header", "true") 
        .option("inferSchema", "true") 
        .option("delimiter", ";") 
        .load(file_path)
)

# COMMAND ----------

# Verificando o Schema importado do arquivo
df.printSchema()

# COMMAND ----------

# Função para ajustar os nomes das colunas
def adjust_column_name(col_name):
    # Converter para maiúsculas
    col_name = col_name.upper()
    # Substituir espaços por underscores e remover caracteres especiais
    col_name = re.sub(r'[^\w\s]', '', col_name)  
    col_name = col_name.replace(" ", "_")  
    return col_name


# COMMAND ----------

# Aplicando a transformação nos nomes das colunas
new_columns = [adjust_column_name(col) for col in df.columns]
df_adjusted = df.toDF(*new_columns)

# COMMAND ----------

# Exibindo o schema ajustado
df_adjusted.printSchema()

# COMMAND ----------

# Remover caracteres não numéricos (como símbolos de moeda) e converter para float
df_adjusted = (
    df_adjusted.withColumn("VALORPARCELA", regexp_replace(col("VALORPARCELA"), "[^0-9.]", "").cast("float")) 
        .withColumn("SALDO_DEVEDOR", regexp_replace(col("SALDO_DEVEDOR"), "[^0-9.]", "").cast("float")) 
        .withColumn("VALOR_BRUTO", regexp_replace(col("VALOR_BRUTO"), "[^0-9.]", "").cast("float")) 
        .withColumn("TROCO", regexp_replace(col("TROCO"), "[^0-9.]", "").cast("float"))
)

# COMMAND ----------

# Exibindo o schema ajustado
df_adjusted.printSchema()

# COMMAND ----------

# Verificando os dados
df_adjusted.display()

# COMMAND ----------

# Caminho para salvar os dados no DBFS
output_path = "dbfs:/FileStore/shared_uploads/matsudrivernf@gmail.com/processed_data"

# COMMAND ----------

# Escrevendo os dados particionados pela coluna DTA_VALIDADE
df_adjusted.write.mode("overwrite").format("parquet").partitionBy("DTA_VALIDADE").save(output_path)
