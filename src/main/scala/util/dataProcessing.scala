package util

import java.util.{TreeMap => JTreeMap}

import scala.annotation.tailrec


object dataProcessing {

  def returnHistoryAndWeight(list:List[(Float,Long)],returnSize:Int) ={
    val padOrCutList = if(list.size==returnSize) {list}
    else if(list.size<returnSize) {List.fill(returnSize-list.length)(0f).++(list.map(_._1))
      .zip(List.fill(returnSize-list.length)(0l).++(list.map(_._2)))}
    else {list.drop(list.length-returnSize)}
    val historyList = padOrCutList.map(_._1)
    val weightList = time2Weight(padOrCutList.map(_._2))
    (historyList,weightList)
  }

  def padOrCut(list:List[Float],padNum:Int,padValue:Float) ={
    if(list.length==padNum) list
    else if(list.length<padNum) list.++(List.fill(padNum-list.length)(padValue))
    else list.drop(list.length-padNum)
  }

  def time2Weight(list:List[Long]) ={
    val list1 = list.map(x=> if(x==0l) {0d} else {1/math.log(x.toDouble/1000/60)})
    val sum = list1.sum
    list1.map(x=>if(sum==0d) {0f} else {math.abs((x/sum).toFloat)})
  }

  def listSplit(list: List[(Float,String,Long)]) ={
    @tailrec
    def listSplitIn(list: List[(Float,String,Long)],returnList:List[(Float,String,Long)]):List[(Float,String,Long)] ={
      if(list.head._2=="transaction") returnList.++(List(list.head))
      else listSplitIn(list.drop(1),returnList.++(List(list.head)))
    }
    listSplitIn(list,List[(Float,String,Long)]())
  }

  def list2JTreeMap(list:List[(Long,List[Float])]):JTreeMap[Long,List[Float]] ={
    val JTreeMap = new JTreeMap[Long,List[Float]]()
    list.map(x=>JTreeMap.put(x._1,x._2))
    JTreeMap
  }

  def list2JTreeMap_(list:List[(Long,Float)]):JTreeMap[Long,Float] ={
    val JTreeMap = new JTreeMap[Long,Float]()
    list.map(x=>JTreeMap.put(x._1,x._2))
    JTreeMap
  }


  def main(args: Array[String]): Unit = {
    val timeList = List(0l,0l,2693338l,1758864l,837681l)
    println(time2Weight(timeList))

  }
}
