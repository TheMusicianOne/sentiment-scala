package org.rubigdata

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

object RUBigDataApp {
  def main(args: Array[String]) {
    val base = "hdfs:///cc-crawl/segments/1618039594808.94/wet/"
    val fileRange = 0 to 639
    val wetfiles = fileRange.map(i => f"${base}CC-MAIN-20210423131042-20210423161042-${i}%05d.warc.wet.gz")
    val wetfile = s"hdfs:///cc-crawl/segments/1618039594808.94/wet/CC-MAIN-20210423131042-20210423161042-0063*.warc.wet.gz"
    val spark = SparkSession.builder.appName("RUBigDataApp").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val pipeline = PretrainedPipeline("analyze_sentiment", lang = "en")
    import spark.implicits._

    val wets = wetfiles.map { path =>
    spark.read
    .format("org.rubigdata.warc")
    .load(path)
    .filter($"warcType" === "conversion")
    .select($"warcTargetUri", $"warcBody".as("text"))
    }.reduce(_ union _)

     // Filter for latin script text samples using regex on first 20 chars
    val latinRegex = "^[\\p{IsLatin}0-9\\s.,!?;:'\"()\\[\\]{}\\-_/@#&*\\\\]+$"
    val filtered_wets = wets.filter(substr(col("text"),lit(1),lit(20)).rlike(latinRegex))
    // Split text into sentences by punctuation or newline, filter short sentences, lowercase
    val sentencesDF = filtered_wets
      .select(explode(split(col("text"), "(?<=\\.|!|\\?)\\s+|\\n|\\r|\\t")).alias("sentence"))
      .withColumn("text", lower(trim(col("sentence"))))
      .filter(length(col("sentence")) > 4)
      .select("text")

     // Define target words (substring matching)
    val target_words = List("palestin", "jew", "israel", "west bank")

    // Generate SQL query filtering sentences containing any target word
    def generateQuery(words: List[String]): String = {
      val conditions = words.map(word => s"(text LIKE '%${word}%')")
      s"SELECT text FROM sentencesDF WHERE ${conditions.mkString(" OR ")}"
    }
    sentencesDF.createOrReplaceTempView("sentencesDF")
    val query = generateQuery(target_words)
    val contains_sentences = spark.sql(query)

    // Broadcast target words as a Spark array literal
    val targetWordsLit = array(target_words.map(lit): _*)

    // Run sentiment pipeline on filtered sentences
    val withSentiment = pipeline.transform(contains_sentences)

    // Extract matched target words using Spark SQL expression and filter sentences with matches
    val withMatches = withSentiment.withColumn(
      "matched_targets",
      expr(s"""filter(${targetWordsLit.expr.sql}, word -> lower(text) LIKE concat('%', word, '%'))""")
    ).filter(size($"matched_targets") > 0)

    // Explode matched target words and get sentiment results
    val exploded = withMatches
      .withColumn("target_word", explode($"matched_targets"))
      .withColumn("sentiment", $"sentiment.result"(0))

    // Group by target word and sentiment, count occurrences
    val result = exploded.groupBy("target_word", "sentiment").count()
   // val outputPath = "file:///opt/hadoop/rubigdata/result"
    //result.write.mode("overwrite").parquet(outputPath)
    result.show(truncate=false)

    spark.stop()
  }
}

