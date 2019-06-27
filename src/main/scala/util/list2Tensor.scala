package util

import com.intel.analytics.bigdl.tensor.Tensor



object list2Tensor {
  def list2DenseTensor(seq:Seq[Float]) ={
    val dTensor = Tensor[Float](seq.length).fill(0)
    for(i<-seq.indices){
      dTensor.setValue(i+1,seq(i))
    }
    dTensor
  }
  def list2SparseTensor(seq:Seq[Float],dim:Int) ={
    val indices = seq.map(_.toInt).toArray
    val values = Array.fill(indices.size)(1f)
    val shape = Array(dim)
    Tensor.sparse(Array(indices), values, shape)
  }

  def main(args: Array[String]): Unit = {
    val a = Array(3f,5f,11f)
    println(list2DenseTensor(a))
    println(list2SparseTensor(a,20))



  }
}
