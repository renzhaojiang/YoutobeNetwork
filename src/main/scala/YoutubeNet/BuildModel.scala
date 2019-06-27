package YoutubeNet

import com.intel.analytics.bigdl.nn.{BatchNormalization, MM, Mean, SoftMax}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Embedding, Input, KerasLayerWrapper, Merge, Narrow, Reshape}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

object BuildModel {

  def buildModel(modelParams:NetParams) = {
    val embeddingSizeArray = modelParams.embeddingSizeArray
    val embeddingWeightSizeArray = modelParams.embeddingWeightSizeArray
    val itemDim = modelParams.itemDim
    val itemEmbeddingSize = modelParams.itemEmbeddingSize
    val categoryDimArray = modelParams.categoryDimArray
    val categoryEmbeddingSize = modelParams.categoryEmbeddingSize
    val hiddenLayerArray = modelParams.hiddenLayerArray
    //init params
    val viewedSize = embeddingSizeArray(0)
    val addedSize = embeddingSizeArray(1)
    val tradedSize = embeddingSizeArray(2)
    val itemSampleSize = embeddingSizeArray(3)

    val viewedWeightSize = embeddingWeightSizeArray(0)
    val addedWeightSize = embeddingWeightSizeArray(1)
    val tradedWeightSize = embeddingWeightSizeArray(2)
    //data input
    val embeddingInput = Input[Float](Shape(embeddingSizeArray.sum))
    val embeddingWeightInput = Input[Float](Shape(embeddingWeightSizeArray.sum))
    val categoryInput = Input[Float](Shape(categoryDimArray.length))
    //item embedding space
    val itemEmbedding = Embedding[Float](itemDim, itemEmbeddingSize,init = "normal")
      .setName("itemEmbedding").inputs(embeddingInput)
    //split embedding vector
    val viewedEmbedding = Narrow[Float](dim = 1, offset = 0, length = viewedSize).inputs(itemEmbedding)
    val addedEmbedding = Narrow[Float](dim = 1, offset = viewedSize - 1, length = addedSize).inputs(itemEmbedding)
    val tradedEmbedding = Narrow[Float](dim = 1, offset = viewedSize+addedSize-1, length = tradedSize).inputs(itemEmbedding)
    val itemSampleEmbedding = Narrow[Float](dim = 1, offset = viewedSize+addedSize+tradedSize-1, length = itemSampleSize).inputs(itemEmbedding)
    //embedding weight
    var viewedWeight = Narrow[Float](dim = 1, offset = 0, length = viewedWeightSize).inputs(embeddingWeightInput)
    viewedWeight = Reshape[Float](targetShape = Array(1,viewedWeightSize)).inputs(viewedWeight)

    var addedWeight = Narrow[Float](dim = 1, offset = viewedWeightSize-1, length = addedWeightSize).inputs(embeddingWeightInput)
    addedWeight = Reshape[Float](targetShape = Array(1,addedWeightSize)).inputs(addedWeight)

    var tradedWeight = Narrow[Float](dim = 1, offset = viewedWeightSize+addedWeightSize-1, length = tradedWeightSize).inputs(embeddingWeightInput)
    tradedWeight = Reshape[Float](targetShape = Array(1,tradedWeightSize)).inputs(tradedWeight)
    //compute the embedding input
    var viewedVector = new KerasLayerWrapper[Float](MM[Float]()).inputs(Array(viewedWeight,viewedEmbedding))
    viewedVector = Reshape[Float](targetShape = Array(itemEmbeddingSize)).inputs(viewedVector)

    var addedVector = new KerasLayerWrapper[Float](MM[Float]()).inputs(Array(addedWeight,addedEmbedding))
    addedVector = Reshape[Float](targetShape = Array(itemEmbeddingSize)).inputs(addedVector)

    var tradedVector = new KerasLayerWrapper[Float](MM[Float]()).inputs(Array(tradedWeight,tradedEmbedding))
    tradedVector = Reshape[Float](targetShape = Array(itemEmbeddingSize)).inputs(tradedVector)

    var categoryEmbedding = Embedding[Float](categoryDimArray.sum, categoryEmbeddingSize).inputs(categoryInput)
    categoryEmbedding = new KerasLayerWrapper[Float](Mean[Float](dimension = 2)).inputs(categoryEmbedding)
    //concat the embedding input as dnn first layer
    val firstLayer = Merge[Float](mode = "concat").inputs(Array(viewedVector, addedVector, tradedVector, categoryEmbedding))
    //dnn relu,L2
    var dnn = firstLayer
    for (neuronsNum <- hiddenLayerArray) {
      //      dnn = Dense(outputDim = neuronsNum, activation = "relu", wRegularizer = new L1L2Regularizer(0d, 0.05d)).inputs(dnn)
      dnn = Dense[Float](outputDim = neuronsNum, activation = "relu").inputs(dnn)
      dnn = new KerasLayerWrapper[Float](BatchNormalization[Float](neuronsNum)).inputs(dnn)

    }
    dnn = Reshape[Float](targetShape = Array(itemEmbeddingSize, 1)).setName("userVector").inputs(dnn)
    //dnn output as user vector mm with item embedding vector as the result
    var output = new KerasLayerWrapper[Float](MM[Float]()).inputs(Array(itemSampleEmbedding, dnn))
    output = Reshape[Float](targetShape = Array(itemSampleSize)).inputs(output)
    output = new KerasLayerWrapper[Float](SoftMax[Float]()).inputs(output)

    (Model[Float](Array(embeddingInput, embeddingWeightInput, categoryInput), output),
      Model[Float](Array(embeddingInput, embeddingWeightInput, categoryInput), dnn))
  }


}
