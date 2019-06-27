package util

object categoryMerge {
  def merge(categoryList:List[Float],categoryDimArray:Array[Int]) ={
    assert(categoryList.length == categoryDimArray.length)
    val result = categoryList.toBuffer
    var dim = 0
    for(i<-categoryList.indices){
      if(i==0){}
      else {
        result(i) =result(i)+categoryDimArray(i-1)+dim
        dim = dim+categoryDimArray(i-1)
      }
    }
    result.toList
  }
}
