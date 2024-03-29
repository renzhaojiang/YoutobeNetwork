package YoutubeNet.recall

import com.intel.analytics.bigdl.dataset.TensorSample
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import util.categoryMerge
import util.list2Tensor.list2DenseTensor
import YoutubeNet.recall.FeatureEngine.merge

object serving {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val conf = Engine.createSparkConf()
  val session = SparkSession.builder.config(conf).master("local[2]").appName("YoutubeNet").getOrCreate()
  Engine.init

  def loadUserVectorModel(path:String)    = {
    Model.getClass()  //scala static block call
    val model = Module.loadModule[Float](path)
    println(model.getInputShape(),model.getOutputShape())
    LocalPredictor(model)
  }

  def mapUserFeature() = {
    val itemSize = 8
    val labelName = "transaction"
    val historyBehaviorNameArray = Array("view","addtocart","transaction")
    val historyBehaviorReturnSizeArray = Array(50,50,30)
    val (mergeRDD,itemDim) = merge(session,itemSize,labelName,historyBehaviorNameArray,historyBehaviorReturnSizeArray)
    val categoryDimArray = Array(mergeRDD.map(_._2._3.head).max.toInt+1,mergeRDD.map(_._2._3.last).max.toInt+1)
    val userItemSample = mergeRDD.map{case ((user,item,timestamp,futureItemList),(itemInput,embeddingInput,embeddingWeight,categoryList,label))=>{
      val itemInputTensor = list2DenseTensor(itemInput)
      val embeddingTensor = list2DenseTensor(embeddingInput)
      val embeddingWeightTensor = list2DenseTensor(embeddingWeight)
      val categoryTensor = list2DenseTensor(categoryMerge.merge(categoryList,categoryDimArray))
      val labelTensor = list2DenseTensor(label).reshape(Array(itemSize))
      ((user,item,timestamp,futureItemList),TensorSample[Float](Array(itemInputTensor,embeddingTensor,embeddingWeightTensor,categoryTensor),Array(labelTensor)))
    }}
    userItemSample
  }

  def main(args: Array[String]): Unit = {
    val userVectorModel = loadUserVectorModel("..\\data\\params\\userVectorModel\\model")

    val mergeRDD = mapUserFeature
    mergeRDD.take(10).foreach(println)

    val user = mergeRDD.map(_._2).take(1)

    val userVector = userVectorModel.predict(user).head.asInstanceOf[Tensor[Float]].toArray().toList
    println(userVector)

  }
}
