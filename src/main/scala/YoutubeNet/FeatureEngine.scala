package YoutubeNet

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import util.dataProcessing.{list2JTreeMap, returnHistoryAndWeight}
import util.negativeSampling.{buildTable, getNegativeSample}

object FeatureEngine {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def loadEventData(session: SparkSession):(RDD[(Float,((List[Float],List[Float]),(List[Float],List[Float]),(List[Float],List[Float]),Float,Long))],Array[Float]) ={
    val window = 259200000l //3days

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

    val abnormalUserList = eventRDD.filter(_._4 == "transaction").map(_._2).countByValue().filter(_._2>20).keys.toSet
    val unpopularItem = eventRDD.map(_._3).countByValue().filter(_._2<=2).keys.toSet

    eventRDD = eventRDD.filter(x=>{!abnormalUserList.contains(x._2) && !unpopularItem.contains(x._3)})

    val userTradedItemTimestampMap = eventRDD.filter(_._4 == "transaction").map(x=>(x._2, List((x._1,x._3))) ).reduceByKey(_++_)
      .collectAsMap()

    val eventRDDAfter = eventRDD.map{case (timestamp,user,item,event)=>(user, List((item,event,timestamp)) )}.reduceByKey(_++_)
      .filter(_._2.map(_._2).contains("transaction"))
      .map{case (user,list)=>{
        val transactionItemList = list.filter(_._2=="transaction")
        val history = transactionItemList.map{case (transactionItem,_,transactionTimestamp)=>{
          val viewedItemList = list.filter(e=>{e._2=="view" && transactionTimestamp-e._3>0l && transactionTimestamp-e._3<=window})
            .map{case (viewedItem,_,viewedTimestamp)=>(viewedItem,transactionTimestamp-viewedTimestamp)}
            .sortWith(_._2>_._2)
          val viewedItemListAfter = returnHistoryAndWeight(viewedItemList,20)

          val addItemList = list.filter(e=>{e._2=="addtocart" && transactionTimestamp-e._3>0l && transactionTimestamp-e._3<=window})
            .map{case (addItem,_,addTimestamp)=>(addItem,transactionTimestamp-addTimestamp)}
            .sortWith(_._2>_._2)
          val addItemListAfter = returnHistoryAndWeight(addItemList,10)

          val tradedItemList = userTradedItemTimestampMap(user).filter(x=>{x._1<transactionTimestamp})
            .map{case (tradedTimestamp,tradedItem)=>(tradedItem,transactionTimestamp-tradedTimestamp)}
            .sortWith(_._2>_._2)
          val tradedItemListAfter = returnHistoryAndWeight(tradedItemList,10)

          (viewedItemListAfter,addItemListAfter,tradedItemListAfter,transactionItem,transactionTimestamp)
        }}
        (user,history)
      }
      }
      .flatMapValues(e=>e)

    val itemCount = eventRDDAfter.map(_._2._4).countByValue().toMap
    val table = buildTable(itemCount)

    (eventRDDAfter,table)
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
    val (eventRDD,table)= loadEventData(session)
    //    eventRDD.take(10).foreach(println)
    val itemCategoryMap = loadCategoryData(session)
    //    itemCategoryMap.take(10).foreach(println)

    val mergeRDD = eventRDD.map{case (user,(viewed,added,traded,item,timestamp))=>{
      val categoryList:List[Float]= if(itemCategoryMap.contains(item)) {
        if(itemCategoryMap(item).lowerKey(timestamp)!=null){
          itemCategoryMap(item).lowerEntry(timestamp).getValue} else itemCategoryMap(item).firstEntry().getValue}
      else List(0f,0f)
      val tradedItemList = traded._1
      val itemList = List(item)++getNegativeSample(table,3,tradedItemList)
      val label =  List(1f)++Array.fill(itemList.size-1){0f}.toList
      (user,item,timestamp,(viewed._1++added._1++traded._1++itemList,viewed._2++added._2++traded._2,categoryList,label))
    }
    }
    mergeRDD
  }
}
