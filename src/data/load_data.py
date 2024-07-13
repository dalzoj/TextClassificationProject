import pandas as pd
import src.utils.general_path as general_path

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


def main():
    sc = SparkContext.getOrCreate(SparkConf().setMaster('local[*]'))
    spark = (SparkSession.builder.getOrCreate())
    balanced_dfs = spark.read.option('multiLine', 'true').option('quotechar', '"').option('inferSchema', 'true').csv(general_path.RAW_SPARK_DATA_PATH+'*.csv', sep = '\u0001').toPandas()

    select_data = balanced_dfs[['_c2','_c3']]
    select_data = select_data.rename(
        columns = {
            '_c2':'label',
            '_c3':'text'}
        )

    select_data.to_csv(
        general_path.RAW_DATA_PATH + 'analytic_data.csv',
        index = False
        )


if __name__ == "__main__":
    main()