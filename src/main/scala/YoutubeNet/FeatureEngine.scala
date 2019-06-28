package YoutubeNet

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import util.dataProcessing.{historyList2Feature, list2JTreeMap, returnHistoryAndWeight}
import util.negativeSampling.{buildTable, getNegativeSample}

object FeatureEngine {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def loadEventData(session: SparkSession) ={
    val windowTime = 259200000l //3days

    var eventRDD:RDD[(Long,Float,Float,String)] = session.sparkContext.textFile("..\\data\\EcommerceDataSet\\events.csv")
      .mapPartitionsWithIndex {
        (idx, iter) => if (idx == 0) iter.drop(1) else iter
      }
      .map(x=>{
        val line = x.split(",")
        val timestamp = line(0).toLong
        val user = line(1).toFloat
        val item = line(3).toFloat+1f
        val event = line(2)
        (timestamp,user,item,event)
      })

    //filter useless user and unpopular item
    val uselessUserList = eventRDD.map(_._2).countByValue().filter(_._2<2f).keys.toSet
    val unpopularItem = eventRDD.map(_._3).countByValue().filter(_._2<5f).keys.toSet
    eventRDD = eventRDD.filter(x=>{!uselessUserList.contains(x._2) && !unpopularItem.contains(x._3)})

    //change item to index
    val itemIndexMap:Map[Float,Int] = eventRDD.map(_._3).distinct().collect().toList.zipWithIndex.toMap
    val itemDim:Int = itemIndexMap.size+1
    eventRDD = eventRDD.map{case (timestamp,user,item,event)=> (timestamp,user,itemIndexMap(item)+1f,event)}
    //extract history behavior
    val userSampleRDD = eventRDD.map{case (timestamp,user,item,event)=>(user, List((timestamp,event,item)) )}
      .reduceByKey(_++_).filter(_._2.map(_._2).contains("transaction"))
      .flatMapValues(historyList2Feature(_,windowTime,"transaction",Array("view","addtocart","transaction"),Array(10,5,3)))
      .map{case (user,(timestamp,item,futureItemList,historyBehavior))=>((user,item,timestamp,futureItemList),historyBehavior)}
    //negative sample table
    val itemCount = eventRDD.map(_._3).countByValue().toMap
    val table = buildTable(itemCount)

    (userSampleRDD,table,itemDim)
  }

  def loadCategoryData(session: SparkSession):Map[Float,java.util.TreeMap[Long,scala.List[Float]]] ={
    val categoryDataRDD1 = session.sparkContext.textFile("..\\data\\EcommerceDataSet\\item_properties_part1.csv")
      .mapPartitionsWithIndex {
        (idx, iter) => if (idx == 0) iter.drop(1) else iter
      }
      .map(x=>{
        val line = x.split(",")
        val timestamp = line(0).toLong
        val item = line(1).toFloat
        val property = line(2)
        val value = line(3)
        (timestamp,item,property,value)
      })
      .filter(_._3 == "categoryid")

    val categoryDataRDD2 = session.sparkContext.textFile("..\\data\\EcommerceDataSet\\item_properties_part2.csv")
      .mapPartitionsWithIndex {
        (idx, iter) => if (idx == 0) iter.drop(1) else iter
      }
      .map(x=>{
        val line = x.split(",")
        val timestamp = line(0).toLong
        val item = line(1).toFloat
        val property = line(2)
        val value = line(3)
        (timestamp,item,property,value)
      })
      .filter(_._3 == "categoryid")

    val categoryParentMap = session.sparkContext.textFile("..\\data\\EcommerceDataSet\\category_tree.csv")
      .mapPartitionsWithIndex {
        (idx, iter) => if (idx == 0) iter.drop(1) else iter
      }
      .map(x=>{
        val line = x.split(",")
        val (category,parent) = if(line.length==1) {(line.head,0f)} else {(line(0),line(1).toFloat)}
        (category.toFloat,parent)
      })
      .collectAsMap()

    val categoryDataRDD = categoryDataRDD1.union(categoryDataRDD2)
      .map{case (timestamp,item,property,value)=>(item,List((timestamp,value.toFloat,categoryParentMap.getOrElse[Float](value.toFloat,0f))) )}
      .reduceByKey(_++_)
      .mapValues(x=>x.map{case (timestamp,category,parent)=>(timestamp,List(category,parent))})
      .mapValues(list2JTreeMap(_))
      .collectAsMap()

    categoryDataRDD.toMap
  }

  def merge(session: SparkSession) ={
    val (userSampleRDD,table,itemDim)= loadEventData(session)
    //    eventRDD.take(10).foreach(println)
    val itemCategoryMap = loadCategoryData(session)
    //    itemCategoryMap.take(10).foreach(println)

    val mergeRDD = userSampleRDD.map{case ((user,item,timestamp,futureItemList),historyBehavior)=>{
      val categoryList:List[Float]= if(itemCategoryMap.contains(item)) {
        if(itemCategoryMap(item).lowerKey(timestamp)!=null){
          itemCategoryMap(item).lowerEntry(timestamp).getValue} else itemCategoryMap(item).firstEntry().getValue}
      else List(0f,0f)
      val beforeItemList = historyBehavior.map(_._1).reduce(_++_).distinct++List(item)
      val itemList = List(item)++getNegativeSample(table,7,beforeItemList)
      val label =  List(1f)++Array.fill(itemList.size-1){0f}.toList
      ((user,item,timestamp,futureItemList),(historyBehavior.map(_._1).reduce(_++_)++itemList,historyBehavior.map(_._2).reduce(_++_),categoryList,label))
    }
    }
    (mergeRDD,itemDim)
  }
}
