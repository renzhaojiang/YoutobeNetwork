import YoutubeNet.recall.BuildModel.buildModel
import YoutubeNet.recall.FeatureEngine._
import YoutubeNet.recall.{CompileParams, FitModel, NetworkParams}
import com.intel.analytics.bigdl.dataset.TensorSample
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import util.categoryMerge
import util.list2Tensor.list2DenseTensor

import scala.collection.mutable.ArrayBuffer


object TestYoutubeNet {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val conf = Engine.createSparkConf()
  val session = SparkSession.builder.config(conf).master("local[2]").appName("YoutubeNet").getOrCreate()
  Engine.init

  def main(args: Array[String]): Unit = {
    val itemSize = 8
    val labelName = "transaction"
    val historyBehaviorNameArray = Array("view","addtocart","transaction")
    val historyBehaviorReturnSizeArray = Array(50,50,30)

    val (mergeRDD,itemDim) = merge(session,itemSize,labelName,historyBehaviorNameArray,historyBehaviorReturnSizeArray)
    mergeRDD.take(10).foreach(println)

    val categoryDimArray = Array(mergeRDD.map(_._2._4.head).max.toInt+1,mergeRDD.map(_._2._4.last).max.toInt+1)

    val itemEmbeddingSize = 256
    val historyItemSizeArray = Array(50,50,30)
    val categoryEmbeddingSize = 64
    val hiddenLayerArray = Array(1024,512,itemEmbeddingSize)
    val continuousSize = 0
    val  modelParams = NetworkParams(itemDim,itemEmbeddingSize,itemSize,historyItemSizeArray,categoryDimArray,categoryEmbeddingSize,continuousSize,hiddenLayerArray)
    val (model,userVectorOutputModel) = buildModel(modelParams)

    val lr = 0.0005
    val lrDecay = 1e-6
    val batchSize = 256
    val epoch = 10
    val compileParams = CompileParams(lr,lrDecay,batchSize,epoch)

    val userItemSample = mergeRDD.map{case ((user,item,timestamp,futureItemList),(itemInput,embeddingInput,embeddingWeight,categoryList,label))=>{
      val itemInputTensor = list2DenseTensor(itemInput)
      val embeddingTensor = list2DenseTensor(embeddingInput)
      val embeddingWeightTensor = list2DenseTensor(embeddingWeight)
      val categoryTensor = list2DenseTensor(categoryMerge.merge(categoryList,categoryDimArray))
      val labelTensor = list2DenseTensor(label).reshape(Array(itemSize))
      ((user,item,timestamp,futureItemList),TensorSample[Float](Array(itemInputTensor,embeddingTensor,embeddingWeightTensor,categoryTensor),Array(labelTensor)))
    }}

    FitModel.fit(userItemSample,model,compileParams)

    //save userVectorOutputModel
    userVectorOutputModel.saveModule("..\\data\\params\\userVectorModel\\model",null,true)
    //save itemEmbedding
    val embeddingMatrix = model.getSubModules().filter(_.getName() == "itemEmbedding").head.getWeightsBias()(0)
    println(embeddingMatrix.size())
    println("***********************************")
    println(embeddingMatrix.select(1, 1).toArray().toList)
    println(userVectorOutputModel.getSubModules().filter(_.getName() == "itemEmbedding").head.getWeightsBias()(0).select(1, 1).toArray().toList)
    println("***********************************")
    var itemEmbeddingVector = new ArrayBuffer[String]()
    for(index<-1 to embeddingMatrix.size(1)){
      val row = List(index).++(embeddingMatrix.select(1,index).toArray().toList).mkString(" ")
      itemEmbeddingVector+=row
    }
    itemEmbeddingVector.take(10).foreach(println)
    session.sparkContext.parallelize(itemEmbeddingVector,1)
      .saveAsTextFile("..\\data\\params\\itemEmbedding")
  }

}
