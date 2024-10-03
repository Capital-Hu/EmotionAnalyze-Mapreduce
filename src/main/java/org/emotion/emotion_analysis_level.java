package org.emotion;

import java.util.logging.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.hbase.*;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import com.huaban.analysis.jieba.JiebaSegmenter;

public class emotion_analysis_level {

    private static final Logger LOGGER = Logger.getLogger(emotion_analysis_level.class.getName());


    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        String[] otherArgs = (new GenericOptionsParser(conf, args)).getRemainingArgs();
        if(otherArgs.length < 2) {
            System.err.println("Usage: Emotion_Analysis <in> [<in>...] <out>");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "Emotion Analysis");
        job.setJarByClass(emotion_analysis_level.class);
        job.setMapperClass(emotion_analysis_level.TokenizerMapper.class);
        //job.setCombinerClass(Emotion_Analysis.IntSumReducer.class);
        job.setCombinerClass(emotion_analysis_level.IntSumCombiner.class);
        job.setReducerClass(emotion_analysis_level.IntSumReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job, new Path("hdfs://192.168.10.100:9000/test/movie_comments.csv"));
        //设置文件输出
        FileOutputFormat.setOutputPath(job, new Path("hdfs://192.168.10.100:9000/outputs"));
        //解决输出路径已经存在的问题
        FileSystem fileSystem = FileSystem.get(conf);
        Path outputPath = new Path("hdfs://192.168.10.100:9000/outputs");
        if (fileSystem.exists(outputPath)) {
            fileSystem.delete(outputPath, true);
        }
        //3.执行
        job.waitForCompletion(true);
    }

    public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        public static List<String> most = new ArrayList<>();
        public static List<String> very = new ArrayList<>();
        public static List<String> more = new ArrayList<>();
        public static List<String> ish = new ArrayList<>();
        public static List<String> insufficiently = new ArrayList<>();
        public static List<String> over = new ArrayList<>();
        public static List<String> negative_words = new ArrayList<>();
        public static List<String> postive_words = new ArrayList<>();
        public static List<String> stop_words = new ArrayList<>();
        public static double score(List<String> words)
        {
            System.out.println("words:"+words);
            double most_score=8;
            double very_score=6;
            double more_score=4;
            double ish_score=0.6;
            double insufficiently_score= -1.5;
            double over_score=2;
            double postive_score=1;
            double negative_score=-1;
            double no_attitide_score=0;
            List<Double>PS=new ArrayList<>();
            List<Double>NS=new ArrayList<>();
            List<Double>NAS=new ArrayList<>();
            for (int i = 0; i < words.size(); i++)
            {
                if (negative_words.contains(words.get(i)))
                {
                    if(i==0)
                    {
                        PS.add(negative_score);
                    }
                    else {
                        if(most.contains(words.get(i-1)))
                            NS.add(most_score*negative_score);
                        else if (very.contains(words.get(i-1)))
                            NS.add(very_score*negative_score);
                        else if (more.contains(words.get(i-1)))
                            NS.add(more_score*negative_score);
                        else if (ish.contains(words.get(i-1)))
                            NS.add(ish_score*negative_score);
                        else if (insufficiently.contains(words.get(i-1)))
                            NS.add(insufficiently_score*negative_score);
                        else if (over.contains(words.get(i-1)))
                            NS.add(over_score*negative_score);
                        else
                            NS.add(negative_score);
                    }
                }
                else if (postive_words.contains(words.get(i)))
                {
                    if(i==0)
                    {
                        PS.add(postive_score);
                    }
                    else {
                        if(most.contains(words.get(i-1)))
                            PS.add(most_score*postive_score);
                        else if (very.contains(words.get(i-1)))
                            PS.add(very_score*postive_score);
                        else if (more.contains(words.get(i-1)))
                            PS.add(more_score*postive_score);
                        else if (ish.contains(words.get(i-1)))
                            PS.add(ish_score*postive_score);
                        else if (insufficiently.contains(words.get(i-1)))
                            PS.add(insufficiently_score*postive_score);
                        else if (over.contains(words.get(i-1)))
                            PS.add(over_score*postive_score);
                        else
                            PS.add(postive_score);
                    }
                }
                else {
                    NAS.add(no_attitide_score);
                }
            }
            System.out.println("PS:"+PS);
            System.out.println("NS:"+NS);
            System.out.println("NAS:"+NAS);
            Double NS_sum = NS.stream().reduce(Double::sum).orElse(0.0);
            Double PS_sum = PS.stream().reduce(Double::sum).orElse(0.0);
            System.out.println("NS_sum:"+NS_sum);
            System.out.println("PS_sum:"+PS_sum);
            double final_score=NS_sum+PS_sum;
            return final_score;
        }
        public static int fenfen(double b) {
            int a=(int)b;
            int level=0;
            if (a<0) {
                level=a/5-1;
            }
            else if (a>0) {
                level=a/5+1;
            }
            else if (a==0) {
                return level;
            }
            return level;
        }
        public static boolean isNumeric(String str){
            Pattern pattern = Pattern.compile("[0-9]*");
            Matcher isNum = pattern.matcher(str);
            if( !isNum.matches() ){
                return true;
            }
            return false;
        }
        public static void read() throws IOException {

            Configuration conf = new Configuration();
            FileSystem fs = FileSystem.get(URI.create("hdfs://192.168.10.100:9000"), conf);

//            String chinese_degree_path = "hdfs://192.168.10.100:9000/wordsLib/degree.txt";
            BufferedReader degreefile = new BufferedReader(new InputStreamReader(fs.open(new Path("/wordsLib/degree.txt")), "UTF-8"));
            String temp = null;
            most = new ArrayList<>();
            very = new ArrayList<>();
            more = new ArrayList<>();
            ish = new ArrayList<>();
            insufficiently = new ArrayList<>();
            over = new ArrayList<>();
            List<List> eList= new ArrayList<>();
            eList.add(most);
            eList.add(very);
            eList.add(more);
            eList.add(ish);
            eList.add(insufficiently);
            eList.add(over);
            int i=-1;
            while ((temp = degreefile.readLine()) != null) {
                if(temp.contains("1")||temp.contains("2")||temp.contains("3")||temp.contains("4")||temp.contains("5")||temp.contains("6")) {
                    i=i+1;
                    continue;
                }
                eList.get(i).add(temp);
            }

//            String negative_comments_path="hdfs://192.168.10.100:9000/wordsLib/negativeComment.txt";
//            String negative_emotion_path="hdfs://192.168.10.100:9000/wordsLib/negativeEmotion.txt";
//            String postive_comments_path="hdfs://192.168.10.100:9000/wordsLib/postiveComment.txt";
//            String postive_emotion_path = "hdfs://192.168.10.100:9000/wordsLib/postiveEmotion.txt";
            BufferedReader negative_comments_file = new BufferedReader(new InputStreamReader(fs.open(new Path("/wordsLib/negativeComment.txt")), "UTF-8"));
            BufferedReader negative_emotion_file = new BufferedReader(new InputStreamReader(fs.open(new Path("/wordsLib/negativeEmotion.txt")), "UTF-8"));
            BufferedReader postive_comments_file = new BufferedReader(new InputStreamReader(fs.open(new Path("/wordsLib/positiveComment.txt")), "UTF-8"));
            BufferedReader postive_emotion_file = new BufferedReader(new InputStreamReader(fs.open(new Path("/wordsLib/positiveEmotion.txt")), "UTF-8"));
            while ((temp = negative_comments_file.readLine()) != null) {
                negative_words.add(temp.replace(" ", ""));
            }
            while ((temp = negative_emotion_file.readLine()) != null) {
                negative_words.add(temp.replace(" ", ""));
            }
            while ((temp = postive_comments_file.readLine()) != null) {
                postive_words.add(temp.replace(" ", ""));
            }
            while ((temp = postive_emotion_file.readLine()) != null) {
                postive_words.add(temp.replace(" ", ""));
            }

//            String filepath="hdfs://192.168.10.100:9000/wordsLib/stopwords.txt";
//            File file =new File(filepath);
//            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            BufferedReader stop_words_file = new BufferedReader(new InputStreamReader(fs.open(new Path("/wordsLib/stopwords.txt")), "UTF-8"));
            while ((temp = stop_words_file.readLine()) != null) {
                stop_words.add(temp.replace(" ", ""));
            }
            System.out.println("positive:"+postive_words);

        }
        public static List<String> withoutstopwords(String oldstring) throws IOException{
            String newString = oldstring;
//            System.out.println("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
            System.out.println("before jieba:"+newString);
            JiebaSegmenter segmenter = new JiebaSegmenter();
            List<String>termlist=segmenter.sentenceProcess(newString);
            System.out.println("after jieba:"+termlist.toString());
            //System.out.println(termlist);
//            LOGGER.info(termlist.toString());
            termlist.removeAll(stop_words);
            System.out.println("after remove:"+termlist.toString());
            //return newString;
            return termlist;

        }
        public TokenizerMapper() throws IOException {
        }
        public static Configuration configuration;
//        public static Connection connection;
//        public static Admin admin;
//        public static Table table;
//        public static void insertData(String rowKey,String colFamily,String col,String val) throws IOException {
//            Put put = new Put(rowKey.getBytes());
//            put.addColumn(colFamily.getBytes(),col.getBytes(), val.getBytes());
//            table.put(put);
//        }
        public void setup() throws IOException, InterruptedException {
//            System.out.println("setup");
            read();
//            configuration  = HBaseConfiguration.create();
//            configuration.set("hbase.zookeeper.quorum","hadoop100");
//            connection = ConnectionFactory.createConnection(configuration);
//            admin = connection.getAdmin();
//            TableName tableName=TableName.valueOf("movie");
//            String[] colFamily= {"information"};
//            if (admin.tableExists(tableName))
//            {
//                System.out.println("文件存在，我要删了他");
//                admin.disableTable(tableName);
//                admin.deleteTable(tableName);
//            }
//            HTableDescriptor tableDescriptor = new HTableDescriptor(tableName);
//            for(String str:colFamily){
//                tableDescriptor.addFamily(new HColumnDescriptor(str));
//            }
//            admin.createTable(tableDescriptor);
//            table = connection.getTable(TableName.valueOf("movie"));

        }
        public void run(Context context) throws IOException, InterruptedException {
            setup();
            try {
                while (context.nextKeyValue()) {
                    map(context.getCurrentKey(), context.getCurrentValue(), context);
                }
            } finally {
                cleanup(context);
            }
        }

        public void map(LongWritable key, Text value, Mapper<LongWritable, Text, Text, DoubleWritable>.Context context) throws IOException, InterruptedException {



            String line=value.toString();
            System.out.println("raw:"+line);

            String[] words = line.split(",");
            System.out.println("split:"+Arrays.toString(words));
//            LOGGER.info(words[0]);
            if (words.length-1<=1)
                return;
            String[] pre=Arrays.copyOfRange(words,0,1);
            String[] comment_lines=Arrays.copyOfRange(words, 2,words.length);
            StringBuilder commentString= new StringBuilder();
            for(String comment:comment_lines)
            {
                commentString.append(comment);
            }
            if (isNumeric(pre[0]))
            {
                return;
            }


            System.out.println("commentString:"+commentString.toString());
            List<String> comment=withoutstopwords(commentString.toString());
            double most_score=8;
            double very_score=6;
            double more_score=4;
            double ish_score=0.6;
            double insufficiently_score= -1.5;
            double over_score=2;
            double postive_score=1;
            double negative_score=-1;
            double no_attitide_score=0;
            for (int i = 0; i < comment.size(); i++)
            {
                if (negative_words.contains(comment.get(i)))
                {
                    if(i==0)
                    {
                        context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(negative_score));
                    }
                    else {
                        if(most.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(most_score*negative_score));
                        else if (very.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(very_score*negative_score));
                        else if (more.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(more_score*negative_score));
                        else if (ish.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(ish_score*negative_score));
                        else if (insufficiently.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(insufficiently_score*negative_score));
                        else if (over.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(over_score*negative_score));
                        else
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(negative_score));
                    }
                }
                else if (postive_words.contains(comment.get(i)))
                {
                    if(i==0)
                    {
                        context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(postive_score));
                    }
                    else {
                        if(most.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(most_score*postive_score));
                        else if (very.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(very_score*postive_score));
                        else if (more.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(more_score*postive_score));
                        else if (ish.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(ish_score*postive_score));
                        else if (insufficiently.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(insufficiently_score*postive_score));
                        else if (over.contains(comment.get(i-1)))
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(over_score*postive_score));
                        else
                            context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(postive_score));
                    }
                }
                else {
                    context.write(new Text(pre[0]+","+pre[1]),new DoubleWritable(no_attitide_score));
                }
            }
        }
    }


    public static class IntSumCombiner extends Reducer<Text, DoubleWritable, Text, IntWritable> {
        public IntSumCombiner() {
        }
        private Text info = new Text();
        public void reduce(Text key, Iterable<DoubleWritable> values, Reducer<Text, DoubleWritable, Text, IntWritable>.Context context) throws IOException, InterruptedException {
            double sum=0.0;
            for (DoubleWritable value : values){
                sum = sum+value.get();
            }
            String newKey = key.toString().split(",")[0]+",  score:"+ sum;
            info.set(newKey);
            context.write(info, new IntWritable(1));
        }
    }


    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, DoubleWritable> {
        public IntSumReducer() {
        }
        public void reduce(Text key, Iterable<IntWritable> values, Reducer<Text, IntWritable, Text, DoubleWritable>.Context context) throws IOException, InterruptedException {
            int sum = 0;
            IntWritable val;
            for(Iterator i$ = values.iterator(); i$.hasNext(); sum += val.get()) {
                val = (IntWritable)i$.next();
            }
            context.write(key, new DoubleWritable(sum));
        }
    }
}


