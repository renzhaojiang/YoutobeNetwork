package util

import scala.util.Random
import math._

object negativeSampling {
  def buildTable(itemCount:Map[Float,Long]) ={
    val tableSize = itemCount.keys.max.toInt*100
    val table: Array[Float] = new Array(tableSize)
    val itemCountPowerSum = itemCount.mapValues(pow(_,0.75)).values.sum
    val itemCountPow = itemCount.mapValues(pow(_,0.75)/itemCountPowerSum)
    def itemProb(item:Float) = itemCountPow.getOrElse[Double](item,0d)

    var item= 1f
    var index = 0
    var itemProbability = itemProb(item)
    while(index<tableSize) {
      table(index) = item
      if(index / tableSize.toDouble>itemProbability) {
        item += 1f
        itemProbability += itemProb(item)
      }
      index += 1
    }
    table
  }

  def getNegativeSample(table:Array[Float],num:Int,beforeItem:List[Float]): List[Float] = {
    val tableSize = table.size
    var negativeList:List[Float] = List()
    def getItem() = table(Random.nextInt(tableSize))
    for(i<- 0 until num+beforeItem.size) {
      var item = getItem()
      while (negativeList.contains(item)) item = getItem()
      negativeList = negativeList++List(item)
    }
    negativeList.diff(beforeItem).take(num)
  }

  def main(args: Array[String]): Unit = {
    val table = buildTable(Map(2.0f->3l,3.0f->3l,9f->3l,10f->3l))
    var list:List[Float] = List()
    for(i<-0 until 10000) {
      val returnList =getNegativeSample(table,1,List(9f))
      if(returnList.head==9f){println("return 9")}
      list = list++returnList
    }
    val count = list.groupBy(identity).mapValues(_.size)
    println(count)

  }
}
