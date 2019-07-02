package YoutubeNet.recall

import com.intel.analytics.bigdl.nn.{BatchNormalization, MM, Mean, SoftMax}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

object BuildModel {
  def main(args: Array[String]): Unit = {
    val itemDim = 300
    val itemEmbeddingSize = 256
    val itemSize = 4
    val historyItemSizeArray = Array(10,8,6)
    val categoryDimArray = Array(15,40,70)
    val categoryEmbeddingSize = 32
    val continuousSize = 2
    val hiddenLayerArray = Array(1024,512,itemEmbeddingSize)
    val params = NetworkParams(itemDim,itemEmbeddingSize,itemSize,historyItemSizeArray,categoryDimArray,categoryEmbeddingSize,continuousSize,hiddenLayerArray)

    val (model,userVectorModel) =buildModel(params)
    println(model.getInputShape(),model.getOutputShape())
    println(userVectorModel.getInputShape(),userVectorModel.getOutputShape())

  }

  def buildModel(params:NetworkParams) = {
    val itemDim = params.itemDim
    val itemEmbeddingSize = params.itemEmbeddingSize
    val itemSize = params.itemSize
    val historyItemSizeArray = params.historyItemSizeArray
    val categoryDimArray = params.categoryDimArray
    val categoryEmbeddingSize = params.categoryEmbeddingSize
    val continuousSize = params.continuousSize
    val hiddenLayerArray = params.hiddenLayerArray

    //input
    val itemInput = Input[Float](Shape(itemSize))
    val historyItemInput = Input[Float](Shape(historyItemSizeArray.sum))
    val historyWeightInput = Input[Float](Shape(historyItemSizeArray.sum))
    val categoryInput = Input[Float](Shape(categoryDimArray.length))
    val continuousInput = Input[Float](Shape(continuousSize))
    //concat item and history behavior
    val itemEmbeddingInput = Merge[Float](mode = "concat").inputs(Array(itemInput, historyItemInput))
    //item and history behavior embedding with the same embedding space
    val itemEmbeddingOutput = Embedding[Float](itemDim, itemEmbeddingSize,init = "normal").setName("itemEmbedding").inputs(itemEmbeddingInput)
    //split itemVector and historyVector
    val itemVector = Narrow[Float](dim = 1, offset = 0, length = itemSize).inputs(itemEmbeddingOutput)
    val historyItemEmbeddingOutput = Narrow[Float](dim = 1, offset = itemSize, length = historyItemSizeArray.sum).inputs(itemEmbeddingOutput)
    //compute history behavior and concat
    var startIndex = 0
    val embeddingMatrix = Narrow[Float](dim = 1, offset = 0, length = historyItemSizeArray(0)).inputs(historyItemEmbeddingOutput)
    val weight = Reshape[Float](targetShape = Array(1,historyItemSizeArray(0)))
      .inputs(Narrow[Float](dim = 1, offset = 0, length = historyItemSizeArray(0)).inputs(historyWeightInput))
    val historyEmbedding = new KerasLayerWrapper[Float](MM[Float]()).inputs(Array(weight,embeddingMatrix))
    var historyEmbeddingConcat = historyEmbedding
    for(i<-historyItemSizeArray.indices){
      if(i==0){}
      else {
        val embeddingMatrix = Narrow[Float](dim = 1, offset = startIndex, length = historyItemSizeArray(i)).inputs(historyItemEmbeddingOutput)
        val weight = Reshape[Float](targetShape = Array(1,historyItemSizeArray(i)))
          .inputs(Narrow[Float](dim = 1, offset = startIndex, length = historyItemSizeArray(i)).inputs(historyWeightInput))
        val historyEmbedding = new KerasLayerWrapper[Float](MM[Float]()).inputs(Array(weight,embeddingMatrix))
        historyEmbeddingConcat =  Merge[Float](mode = "concat").inputs(Array(historyEmbeddingConcat, historyEmbedding))
      }
      startIndex = startIndex + historyItemSizeArray(i)
    }
    historyEmbeddingConcat = Reshape[Float](targetShape = Array(itemEmbeddingSize*historyItemSizeArray.length)).inputs(historyEmbeddingConcat)
    //category input through embedding and agg
    var categoryEmbedding = Embedding[Float](categoryDimArray.sum, categoryEmbeddingSize).inputs(categoryInput)
    categoryEmbedding = new KerasLayerWrapper[Float](Mean[Float](dimension = 2)).inputs(categoryEmbedding)
    //concat (historyEmbedding,categoryEmbedding,continuousInput) as dnn firstLayer
    var firstLayer = Merge[Float](mode = "concat").inputs(Array(historyEmbeddingConcat, categoryEmbedding))
    if(continuousSize==0){}
    else {firstLayer = Merge[Float](mode = "concat").inputs(Array(firstLayer, continuousInput))}
    //three layers dnn
    var dnn = firstLayer
    for (neuronsNum <- hiddenLayerArray) {
      dnn = Dense[Float](outputDim = neuronsNum, activation = "relu").inputs(dnn)
      dnn = new KerasLayerWrapper[Float](BatchNormalization[Float](neuronsNum)).inputs(dnn)
    }
    dnn = Reshape[Float](targetShape = Array(itemEmbeddingSize, 1)).setName("userVector").inputs(dnn)
    //dnn output as user vector mm with item embedding vector as the result
    var output = new KerasLayerWrapper[Float](MM[Float]()).inputs(Array(itemVector, dnn))
    output = Reshape[Float](targetShape = Array(-1)).inputs(output)
    output = new KerasLayerWrapper[Float](SoftMax[Float]()).inputs(output)

    val inputArray = if(continuousSize==0){
      Array(itemInput, historyItemInput,historyWeightInput, categoryInput)
    }
    else {
      Array(itemInput, historyItemInput,historyWeightInput, categoryInput,continuousInput)
    }

    (Model[Float](inputArray, output),Model[Float](inputArray, dnn))
  }
}
