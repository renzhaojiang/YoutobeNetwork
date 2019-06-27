import YoutubeNet.BuildModel.buildModel
import YoutubeNet.FeatureEngine._
import YoutubeNet.{CompileParams, FitModel, NetParams}
import com.intel.analytics.bigdl.dataset.TensorSample
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import util.categoryMerge
import util.list2Tensor.list2DenseTensor


object TestYoutubeNet {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val conf = Engine.createSparkConf()
  val session = SparkSession.builder.config(conf).master("local[2]").appName("YoutubeNet").getOrCreate()
  Engine.init

  def main(args: Array[String]): Unit = {
    val mergeRDD = merge(session)
    //    mergeRDD.take(10).foreach(println)

    val itemDim = mergeRDD.map(_._4._1).reduce(_++_).toSet.max.toInt+1
    val categoryDimArray = Array(mergeRDD.map(_._4._3.head).max.toInt+1,mergeRDD.map(_._4._3.last).max.toInt+1)

    val embeddingSizeArray = Array(20,10,10,4)
    val embeddingWeightSizeArray = Array(20,10,10)
    val itemEmbeddingSize = 32
    val categoryEmbeddingSize = 16
    val hiddenLayerArray = Array(32,32,itemEmbeddingSize)

    val modelParams = NetParams(embeddingSizeArray,embeddingWeightSizeArray,itemDim,itemEmbeddingSize,categoryDimArray,categoryEmbeddingSize,hiddenLayerArray)
    val (model,userVectorOutputModel) = buildModel(modelParams)
    println(model.getInputShape(),model.getOutputShape())
    println(userVectorOutputModel.getInputShape(),userVectorOutputModel.getOutputShape())

    val lr = 0.0005
    val lrDecay = 1e-6
    val batchSize = 128
    val epoch = 5
    val compileParams = CompileParams(lr,lrDecay,batchSize,epoch)

    val userItemSample = mergeRDD.map{case (user,item,timestamp,(embeddingInput,embeddingWeight,categoryList,label))=>{
      val embeddingTensor = list2DenseTensor(embeddingInput)
      val embeddingWeightTensor = list2DenseTensor(embeddingWeight)
      val categoryTensor = list2DenseTensor(categoryMerge.merge(categoryList,categoryDimArray))
      val labelTensor = list2DenseTensor(label).reshape(Array(4))
      (user,item,timestamp,TensorSample[Float](Array(embeddingTensor,embeddingWeightTensor,categoryTensor),Array(labelTensor)))
    }}

    FitModel.fit(userItemSample,model,compileParams)
  }

}
