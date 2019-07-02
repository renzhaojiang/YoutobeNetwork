package util

import java.util.{TreeMap => JTreeMap}


object dataProcessing {

  def historyList2Feature(historyList:List[(Long,String,Float)]
                          ,windowTime:Long,labelName:String
                          ,historyBehaviorNameArray:Array[String]
                          ,historyBehaviorReturnSizeArray:Array[Int]) ={

    val historyMap = historyBehaviorNameArray.zip(historyBehaviorReturnSizeArray).toMap

    historyList.filter(_._2==labelName).map { case (labelTimestamp,labelEvent,labelItem) => {
      val historyBehavior = historyBehaviorNameArray.map{eventName=> {
        val itemTimeList = historyList.map { case (timestamp, event, item) => (labelTimestamp - timestamp, event, item) }
          .filter(list => {
            list._2 == eventName && list._1 > 0 && list._1 < windowTime
          }).map(e=>(e._3,e._1)).sortWith(_._2>_._2)
        val itemTimeListAfter = returnHistoryAndWeight(itemTimeList,historyMap(eventName))
        itemTimeListAfter
      }}.toList
      val futureLabelItemList = historyList.filter(list=>{list._2==labelName&&list._1>labelTimestamp}).map(_._3)
      (labelTimestamp,labelItem,futureLabelItemList,historyBehavior)
    }}
  }

  def returnHistoryAndWeight(list:List[(Float,Long)],returnSize:Int) ={
    val padOrCutList = if(list.size==returnSize) {list}
    else if(list.size<returnSize) {List.fill(returnSize-list.length)((0f,0l))++list}
    else {list.drop(list.length-returnSize)}

    val historyList = padOrCutList.map(_._1)
    val weightList = time2Weight(padOrCutList.map(_._2))
    (historyList,weightList)
  }

  def padOrCut(list:List[Float],returnSize:Int,padValue:Float) ={
    if(list.length==returnSize) {list}
    else if(list.length<returnSize) {list++List.fill(returnSize-list.length)(padValue)}
    else {list.drop(list.length-returnSize)}
  }

  def time2Weight(list:List[Long]) ={
    val weightList = list.map(x=> if(x==0l) {0d} else {1/math.log(x.toDouble/1000/60+1d)})
    val sum = weightList.sum
    weightList.map(x=>if(sum==0d) {0f} else {math.abs((x/sum).toFloat)})
  }

  def list2JTreeMap1(list:List[(Long,List[Float])]):JTreeMap[Long,List[Float]] ={
    val JTreeMap = new JTreeMap[Long,List[Float]]()
    list.map(x=>JTreeMap.put(x._1,x._2))
    JTreeMap
  }

  def list2JTreeMap2(list:List[(Long,Float)]):JTreeMap[Long,Float] ={
    val JTreeMap = new JTreeMap[Long,Float]()
    list.map(x=>JTreeMap.put(x._1,x._2))
    JTreeMap
  }


  def main(args: Array[String]): Unit = {
    val a= List((1f,2l),(3f,4l))
    println(returnHistoryAndWeight(a,2))

  }
}
