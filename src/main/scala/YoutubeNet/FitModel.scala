package YoutubeNet

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.optim.{Adam, Loss}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.objectives.CategoricalCrossEntropy
import org.apache.spark.rdd.RDD

object FitModel {
  def fit(userItemSample: RDD[((Float, Float, Long,List[Float]), Sample[Float])]
          ,model: Model[Float]
          , compileParams:CompileParams) ={

//    val lr = compileParams.lr
    val lrDecay = compileParams.lrDecay
    val batchSize = compileParams.batchSize
    val epoch = compileParams.epoch

    val sample = userItemSample.map(_._2)
    for(lr<-List(0.0005,0.0003,0.0001)) {
      val Array(trainSampleRDD, valSampleRDD) = sample.randomSplit(Array(0.9, 0.1))
      model.compile(new Adam[Float](learningRate = double2Double(lr)
        , learningRateDecay = double2Double(lrDecay))
        , loss = CategoricalCrossEntropy[Float]()
        , List(new Loss[Float](CategoricalCrossEntropy[Float]())))

      model.fit(trainSampleRDD, int2Integer(batchSize), int2Integer(epoch), validationData = valSampleRDD)
    }
  }

}
