package YoutubeNet

case class NetParams(embeddingSizeArray: Array[Int],
                     embeddingWeightSizeArray: Array[Int],
                     itemDim: Int,
                     itemEmbeddingSize: Int,
                     categoryDimArray: Array[Int],
                     categoryEmbeddingSize: Int,
                     hiddenLayerArray: Array[Int]
                    )
case class CompileParams(lr:Double,
                         lrDecay:Double,
                         batchSize:Int,
                         epoch:Int)


