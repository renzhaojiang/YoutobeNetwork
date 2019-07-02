package YoutubeNet.recall

case class NetworkParams(itemDim :Int,
                         itemEmbeddingSize:Int,
                         itemSize :Int,
                         historyItemSizeArray : Array[Int],
                         categoryDimArray : Array[Int],
                         categoryEmbeddingSize :Int,
                         continuousSize :Int,
                         hiddenLayerArray : Array[Int])

case class CompileParams(lr:Double,
                         lrDecay:Double,
                         batchSize:Int,
                         epoch:Int)


