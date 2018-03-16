# coding: UTF-8
#opencv bgr
#rgb
import numpy as np
import tensorflow as tf
import os
import sys
import csv
import random
from datetime import datetime
from PIL import Image
from matplotlib import pylab as plt
from time import sleep
import cv2
#import cv2.cv as cv

DataShape = (100,100,3)
Ratio = 1#2000.0
TestNum = 100

def show_img(dataDir):

    data = np.load(dataDir)#.astype(float)
    #label = int(row[LabelIdx])
    #data += np.random.normal(0,0.1*np.max(data),(512,512))
    #data -= np.min(data)
    #data = data*255/np.max(data)
    #print name
    #print data.shape
    #if name=="os10-020_cancer_full.npy":
    for i in range(data.shape[0]):
        print (i)
        #lt.imshow(data[i])
        #plt.show()
    #plt.imshow(data[data.shape[0]-ZIdx])
    #plt.show()
    #pilImg = Image.fromarray(np.uint8(data[15]))
    #pilImg.show()
    #sleep(1)
    
def show_npy(dataList):

    #data = np.load(dataDir)#.astype(float)
    #label = int(row[LabelIdx])
    #data += np.random.normal(0,0.1*np.max(data),(512,512))
    #data -= np.min(data)
    #data = data*255/np.max(data)
    #print name
    #print data.shape
    #if name=="os10-020_cancer_full.npy":
    for i in range(len(dataList[0])):
        #print i
        #plt.imshow(dataList[2][i])
        #plt.show()
        pilImg = Image.fromarray(np.uint8(dataList[2][i]*255.0))
        pilImg.show()
        sleep(1)
    #plt.imshow(data[data.shape[0]-ZIdx])
    #plt.show()
    #pilImg = Image.fromarray(np.uint8(data[15]))
    #pilImg.show()
    #sleep(1)
 
# nameList  : list of each data name
# labelList : list of teaching data (written by 0 or 1 and 3 calams)
# dataList  : 4 dimensionary data
def GetData():
    sugi_data = np.load('sugi.npy')/255.0
    hinoki_data = np.load('hinoki.npy')/255.0
    nameList = []
    labelList = []
    dataList = []
    print (sugi_data.shape)
    print (hinoki_data.shape)
    #sys.exit()
    
    repeat = np.min((len(sugi_data),len(hinoki_data)))%100
    for i_repeat in range(repeat):
        for i,data in enumerate(sugi_data[i_repeat*100:(i_repeat+1)*100]):
            name = 'sugi_'+str(i)
            data = np.reshape(data,[DataShape[0],DataShape[1],DataShape[2]])
            label = 0
            l = np.zeros(2)
            l[label] += 1

            labelList.append(l)
            nameList.append([name])
            dataList.append(data)

        print (len(sugi_data))

        for i,data in enumerate(hinoki_data[i_repeat*100:(i_repeat+1)*100]):
            name = 'hinoki_'+str(i)
            data = np.reshape(data,[DataShape[0],DataShape[1],DataShape[2]])
            label = 1
            l = np.zeros(2)
            l[label] += 1

            labelList.append(l)
            nameList.append([name])
            dataList.append(data)
        #print len(hinoki_data)
        #print np.array(dataList).shape
        #print np.max(dataList)
    
    #data_shuffle = zip(nameList,labelList,dataList)
    #random.shuffle(data_shuffle)
    #return zip(*data_shuffle)

    return nameList,labelList,dataList

# dataList : 4 dimensionary data
# testIdxList : 1=test data , 0 = training data
def GetDataset(dataList,testIdxList):
    teDataList = []
    trDataList = []
    for i in range(len(dataList)):
        if i in testIdxList:
            teDataList.append(dataList[i])
        else:
            trDataList.append(dataList[i])
    return np.vstack([teDataList]),np.vstack([trDataList])


def Train(alldata,testIdxList,resultName):
    print ("start train initialize")
    os.mkdir(resultName)
    print ("get dataset")
    ten,trn = GetDataset(alldata[0], testIdxList)
    tel,trl = GetDataset(alldata[1], testIdxList)
    ted,trd = GetDataset(alldata[2], testIdxList)
    #tef,trf = GetDataset(alldata[3], testIdxList)

    #print trd.shape
    #sys.exit()


    #mizumasi 
    #data_num = len(trd)
    #nlist = []
    #llist = []
    #dlist = []
    #flist = []
    #nlist.extend(trn)
    #llist.extend(trl)
    #dlist.extend(trd)
    #flist.extend(trf)
    
    #print data_num
    #label_num = np.zeros(3).astype(np.int32)
    #for i in range(data_num):
    #    for n in range(3):
    #        if trl[i][n]==1 :
    #            label_num[n]+=1
    #print label_num

    #for n in range(3):
    #    for j in range(100-label_num[n]):
    #        flag=1
    #        while(flag==1):
    #          i = random.randint(0,data_num-1)
    #          if trl[i][n]==1:
    #            flag=0  #
    #        nlist.append(trn[i])
    #        llist.append(trl[i])
    #        data = trd[i] + np.random.normal(0,0.1,(DataShape[0],DataShape[1],DataShape[2],1))
    #       dlist.append(data)
            #flist.append(trf[i])

    #for i in range(data_num):
    #    for j in range(10): 
    #        nlist.append(trn[i])
    #        llist.append(trl[i])
    #        data = trd[i] + np.random.normal(0,0.1,(DataShape[0],DataShape[1],DataShape[2],1))
    #        dlist.append(data)
            #flist.append(trf[i])

    #trn = np.array(nlist)
    #trl = np.array(llist)
    #trd = np.array(dlist)
    print ("generate cnn model")
    x = tf.placeholder(tf.float32, [None,DataShape[0],DataShape[1],DataShape[2]])
    y = tf.placeholder(tf.float32, [None,2])

    conv_W0 = tf.Variable(tf.truncated_normal([5,5,3,32],stddev=0.01), name="conv_W0")
    conv_b0 = tf.Variable(tf.zeros([32]), name="conv_b0")
    conv_z0 = tf.nn.conv2d(x, conv_W0, [1,1,1,1], "SAME") + conv_b0
    conv_u0 = tf.nn.relu(conv_z0)
    pool0 = tf.nn.max_pool(conv_u0, [1,2,2,1], [1,2,2,1], "SAME")
    norm0 = pool0#tf.nn.lrn(pool0)   

    conv_W1 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.01), name="conv_W1")
    conv_b1 = tf.Variable(tf.zeros([64]), name="conv_b1")
    conv_z1 = tf.nn.conv2d(norm0, conv_W1, [1,1,1,1], "SAME") + conv_b1
    conv_u1 = tf.nn.relu(conv_z1)
    pool1 = tf.nn.max_pool(conv_u1, [1,2,2,1], [1,2,2,1], "SAME")
    norm1 = pool1#tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

    reshapeVal = 25*25*64
    flat  = tf.nn.dropout(tf.reshape(norm1,[-1,reshapeVal]),0.5)

    full_W0 = tf.Variable(tf.truncated_normal([reshapeVal,200],stddev=0.01), name="full_W0")
    full_b0 = tf.Variable(tf.zeros([200]), name="full_b0")
    full_z0 = tf.matmul(flat, full_W0)  + full_b0
    full_u0 = tf.nn.relu(full_z0)

    full_W = tf.Variable(tf.truncated_normal([200,2],stddev=0.01), name="full_W")
    full_b = tf.Variable(tf.zeros([2]), name="full_b")
    full_z = tf.matmul(full_u0, full_W)  + full_b
    full_u = tf.nn.softmax(full_z)

    #conv_L2 = tf.nn.l2_loss(conv_W0) + tf.nn.l2_loss(full_W)

    loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(full_u,1e-10,1.0))) #+ 0.001*conv_L2
    trainstep = tf.train.AdamOptimizer(1e-5).minimize(loss)
    #trainstep = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    #trainstep = tf.train.FtrlOptimizer(0.0001).minimize(loss)

    collect_prediction = tf.equal(tf.argmax(full_u, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(collect_prediction,tf.float32))
    
    print ("server define")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess, "/home/miyake/Desktop/kahun_ohira/finalmodel.ckpt")

    f = open(resultName+"/LearningResult.csv","w")
    writer = csv.writer(f)
    writer.writerow(["loop","train loss","test accuracy"])
    print ("start learning")

    for i in range(1000001):
    #for i in range(101):
        idxList = []
        for j in range(10):
            idxList.append(random.randint(0,trd.shape[0]-1))

        if i % 10 == 0:
            l = sess.run(loss,feed_dict={x:trd[idxList], y:trl[idxList]})
            acc = sess.run(accuracy,feed_dict={x:ted,y:tel})
            writer.writerow([i,l,acc])
            print (str(i) + ": loss = " + str(l) + ", acc = " + str(acc))
            if l<0.0001 :
                saver.save(sess, datetime.now().strftime('%s')+"finalmodel.ckpt")
                break
        if i % 1000 == 0:
            saver.save(sess, "model.ckpt")

        sess.run(trainstep,feed_dict={x:trd[idxList], y:trl[idxList]})

    f.close()
    print ("output result")
    f = open(resultName+"/TestResult.csv","w")
    writer = csv.writer(f)
    writer.writerow(["Name","IsCollect"])
    isCollect,answer,output,output_data = sess.run((collect_prediction,tf.argmax(y, 1),tf.argmax(full_u, 1),full_u),feed_dict={x:ted,y:tel})
    for i in range(len(output)):
        writer.writerow([ten[i][0],int(isCollect[i]),answer[i],output[i],output_data[i][0],output_data[i][1]])
    tf.reset_default_graph()

def Test(image_path,label):


    limit_data = []
    limit_label = []
    limit_circles = []
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    img_array = np.array(img)
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cimg = cv2.medianBlur(cimg,5)
    #_,cimg = cv2.threshold(cimg,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    #cv2.imwrite(datetime.now().strftime('%s')+"binary.jpg",cimg)
    #sys.exit()

    #circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,10,param1=10,param2=18,minRadius=10,maxRadius=25)
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,10,param1=10,param2=18,minRadius=10,maxRadius=25)
    
    circles = np.uint16(np.around(circles))[0,:]
    print (len(circles))

    for i in circles:
        half = DataShape[0]/2
        zoom_data = img_array[i[1]-half:i[1]+half,i[0]-half:i[0]+half,:]/255.0
        if zoom_data.shape!=DataShape : continue
        limit_data.append(zoom_data)
        l = np.zeros(2)
        l[label] += 1
        limit_label.append(l)
        limit_circles.append(i)
        #cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        #v2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        #print img_array[i[0]-half:i[0]+half,i[1]-half:i[1]+half,:].shape
    limit_data = np.array(limit_data)
    limit_label = np.array(limit_label)
    limit_circles = np.array(limit_circles)
    label_num = limit_data.shape[0]
    #cv2.imwrite(datetime.now().strftime('%s')+"output.jpg",img)
    #sys.exit()

    x = tf.placeholder(tf.float32, [None,DataShape[0],DataShape[1],DataShape[2]])
    y = tf.placeholder(tf.float32, [None,2])

    conv_W0 = tf.Variable(tf.truncated_normal([5,5,3,32],stddev=0.01), name="conv_W0")
    conv_b0 = tf.Variable(tf.zeros([32]), name="conv_b0")
    conv_z0 = tf.nn.conv2d(x, conv_W0, [1,1,1,1], "SAME") + conv_b0
    conv_u0 = tf.nn.relu(conv_z0)
    pool0 = tf.nn.max_pool(conv_u0, [1,2,2,1], [1,2,2,1], "SAME")
    norm0 = pool0#tf.nn.lrn(pool0)   

    conv_W1 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.01), name="conv_W1")
    conv_b1 = tf.Variable(tf.zeros([64]), name="conv_b1")
    conv_z1 = tf.nn.conv2d(norm0, conv_W1, [1,1,1,1], "SAME") + conv_b1
    conv_u1 = tf.nn.relu(conv_z1)
    pool1 = tf.nn.max_pool(conv_u1, [1,2,2,1], [1,2,2,1], "SAME")
    norm1 = pool1#tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

    reshapeVal = 25*25*64
    flat  = tf.nn.dropout(tf.reshape(norm1,[-1,reshapeVal]),0.5)

    full_W0 = tf.Variable(tf.truncated_normal([reshapeVal,200],stddev=0.01), name="full_W0")
    full_b0 = tf.Variable(tf.zeros([200]), name="full_b0")
    full_z0 = tf.matmul(flat, full_W0)  + full_b0
    full_u0 = tf.nn.relu(full_z0)

    full_W = tf.Variable(tf.truncated_normal([200,2],stddev=0.01), name="full_W")
    full_b = tf.Variable(tf.zeros([2]), name="full_b")
    full_z = tf.matmul(full_u0, full_W)  + full_b
    full_u = tf.nn.softmax(full_z)

    #conv_L2 = tf.nn.l2_loss(conv_W0) + tf.nn.l2_loss(full_W)

    loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(full_u,1e-10,1.0))) #+ 0.001*conv_L2
    trainstep = tf.train.AdamOptimizer(1e-5).minimize(loss)
    #trainstep = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    #trainstep = tf.train.FtrlOptimizer(0.0001).minimize(loss)

    collect_prediction = tf.equal(tf.argmax(full_u, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(collect_prediction,tf.float32))
    
    sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "/home/miyake/Desktop/kahun_ohira/finalmodel.ckpt")

    #for i in range(label_num):
    output = sess.run(tf.argmax(full_u, 1),feed_dict={x:limit_data,y:limit_label})
    for i in range(label_num):
        if output[i]==label :
            cv2.circle(img,(circles[i][0],circles[i][1]),circles[i][2],(0,255,0),2)
        else:
            cv2.circle(img,(circles[i][0],circles[i][1]),circles[i][2],(0,0,255),2)

    cv2.imwrite(datetime.now().strftime('%s')+"output.jpg",img)
    sleep(0.1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #f = open(resultName+"/TestResult.csv","w")
    #writer = csv.writer(f)
    #writer.writerow(["Name","IsCollect"])
    #isCollect,answer,output,output_data = sess.run((collect_prediction,tf.argmax(y, 1),tf.argmax(full_u, 1),full_u),feed_dict={x:data,y:label})
    #for i in range(TestNum):
    #    writer.writerow([ten[i][0],int(isCollect[i]),answer[i],output[i],output_data[i][0],output_data[i][1]])
    tf.reset_default_graph()

def learning_data_print(dataList):
    data_num = len(dataList[0])
    for i in range(0,data_num,200):
        for j in range(200):
            path = "image/"+str(i/200)+"/"+ dataList[0][i+j][0] +".png"
            if not os.path.exists(path) :
                plt.imshow(dataList[2][i+j])
                plt.savefig(path)
                plt.clf()
            #pilImg = Image.fromarray(np.uint8(dataList[2][i+j]))
            #pilImg.save("image/"+str(i/200)+"/"+ dataList[0][j][0] +".png")
            #print dataList[2][i]
            #plt.show()

if __name__ == '__main__':
    
    #show_img("/home/miyake/Desktop/kahun_ohira/sugi.npy")
    #Test('/home/miyake/Desktop/kahun_ohira/original_data/test/hinoki/IMG_3782.JPG',1)
    #Test('/home/miyake/Desktop/kahun_ohira/original_data/test/hinoki/IMG_3785.JPG',1)
    #Test('/home/miyake/Desktop/kahun_ohira/original_data/test/hinoki/IMG_3792.JPG',1)
    #Test('/home/miyake/Desktop/kahun_ohira/original_data/test/sugi/IMG_3806.JPG',0)
    #Test('/home/miyake/Desktop/kahun_ohira/original_data/test/sugi/IMG_3808.JPG',0)
    #Test('/home/miyake/Desktop/kahun_ohira/original_data/test/sugi/IMG_3806.JPG',0)

    #Test('/home/miyake/Desktop/kahun_ohira/original_data/mix/IMG_3828.JPG',0)
    #sys.exit()

    alldata = GetData()
    #learning_data_print(alldata)
    #show_npy(alldata)

    data_num = len(alldata[0])
    now = datetime.now()
    for i in range(0,data_num,200):
        testidxList = []
        for j in range(200):
            testidxList.append(i+j)
 
        resultName = now.strftime('%s')+"_Result_"+str(i)+"_"+str(i+200)
        Train(alldata,testidxList,resultName)
        Test('check/20180227･ヒノキ･紫外光･MC(大手町).JPG',1)
        #Test('/home/miyake/Desktop/kahun_ohira/original_data/test/sugi/IMG_3802.JPG',0)
        #Test('/home/miyake/Desktop/kahun_ohira/original_data/mix/IMG_3814.JPG',0)
    