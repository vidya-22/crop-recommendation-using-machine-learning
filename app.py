import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('gradientmodel.pkl', 'rb'))
model2 = pickle.load(open('model.pkl', 'rb'))
model3 = pickle.load(open('logisticmodel.pkl', 'rb'))
model4 = pickle.load(open('decisiontreemodel.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    crop_dict={20:"rice", 11:"maize", 3:"chickpea", 9:"kidneybeans", 18:"pigeonpeas", 13:"mothbeans", 14:"mungbean", 2:"blackgram", 10:"lentil", 19:"pomegranate", 1:"banana", 12:"mango", 7:"grapes", 21:"watermelon", 15:"muskmelon", 0:"apple", 16:"orange", 17:"papaya", 4:"coconut", 6:"cotton", 8:"jute", 5:"Coffee"}
    crop_url={"rice":"https://www.ruralliquidfertilisers.com/wp-content/uploads/2017/01/fast-fact-rice-1.jpg",
               "maize":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRnWQ6eRzMCG3pj95pmbtehUrIZWSOtRiL-dA&usqp=CAU", 
               "chickpea":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrCVUJjBO_02b_LySuchVNpEg1T9L77m-LGg&usqp=CAU", 
               "kidneybeans":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSb6FdYKjAcAuXi_MrswWx7RMMkNZDyHCc9FA&usqp=CAU", 
               "pigeonpeas":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdtLLbfIcLahMREYDQ2HQbt7zuW39XV2Yz-A&usqp=CAU", 
               "mothbeans":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJKtYf5xV5i3v04a2P89KThgo7-00sa3wGsA&usqp=CAU", 
               "mungbean":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRI74MWyeC1cRi1xhDsrvX5UwrEJAvk8xFsRg&usqp=CAU", 
               "blackgram":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTiV3ZTH2ZFkiIVqe39gj7LEj9TXjBOj5t5w&usqp=CAU", 
               "lentil":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0xIbantI8aAnWFPzIwIK8ByQ9TmLn2SxMsw&usqp=CAU", 
               "pomegranate":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHUwR3M_Te1-bwsdEL78N-YeYslMfSKHN4Hw&usqp=CAU", 
               "banana":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQctT7ej9S2SFwAGZm2StUHuxRp6AlCfTAs4w&usqp=CAU", 
               "mango":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT19Sdyr_QFejB7r0rV11s-yw99K78aoQWAug&usqp=CAU", 
               "grapes":"https://economictimes.indiatimes.com/thumb/msid-66115553,width-1200,height-900,resizemode-4,imgsize-961508/gettyimages-842928214.jpg?from=mdr", 
               "watermelon":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZ01wVt8E6i69R6V3IIuU0dACotBPgsP9uMQ&usqp=CAU", 
               "muskmelon":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwJJ8wlm7_F1AxE_858MGS3sPUKR6s7U2LUQ&usqp=CAU", 
               "apple":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQGzaJgtP-mhOw3l_pLv2_sOWlH1I04aoMX4Q&usqp=CAU", 
               "orange":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxxmS2hI2KjKaC18QvdSd3uOjdgrgGpU9E_g&usqp=CAU", 
               "papaya":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ52f04HS0tTWV_BOOXU2mU7ZAJdHUpwxcXfQ&usqp=CAU", 
               "coconut":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWWS_1K93iVkadnJfHkUq8LUSqCrLiFLkcFQ&usqp=CAU", 
               "cotton":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTXefpx4di21GIa5ot8RKDIJYfYaspy8AtnXw&usqp=CAU", 
               "jute":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRJcu8hWBFrXsMsCRrY7Y0XX9hJDeLFTDVWhQ&usqp=CAU", 
               "coffee":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOoW21Ps1t_AycwHtUWuh6YO-G9UHaa8EZnw&usqp=CAU"}
    int_features = [float(x) for x in request.form.values()]
    if(int_features[-1]==1):
        print ("gradient boosting regressor")
        int_features1=int_features[:-1]
        final_features = [np.array(int_features1)]
        prediction = model1.predict(final_features)#gradient boost
        
    elif(int_features[-1]==2):
        print("random forest regressor")
        int_features2=int_features[:-1]
        final_features = [np.array(int_features2)]
        prediction = model2.predict(final_features)#random forest
    elif(int_features[-1]==3):
        print ("logistic regression")
        int_features3=int_features[:-1]
        final_features = [np.array(int_features3)]
        prediction = model3.predict(final_features)
        return render_template('crop_rec.html', prediction_text='{}'.format((prediction[0])), image_url="{}".format(crop_url[prediction[0]]))
    else:
        print("decision tree regressor")
        int_features4=int_features[:-1]
        final_features = [np.array(int_features4)]
        prediction = model4.predict(final_features)
    #output = round(prediction[0], 2)
    result=crop_dict.get(prediction[0]) or crop_dict[min(crop_dict.keys(), key=lambda key: abs(prediction[0]-key))]

    print("The Suggested Crop for Given Climatic Condition is :", result)

    return render_template('crop_rec.html', prediction_text='{}'.format((result)), image_url="{}".format(crop_url[result]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
@app.route('/crop_details')
def crop_details():
    
    crop_url={"rice":"https://www.ruralliquidfertilisers.com/wp-content/uploads/2017/01/fast-fact-rice-1.jpg",
               "maize":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRnWQ6eRzMCG3pj95pmbtehUrIZWSOtRiL-dA&usqp=CAU", 
               "chickpea":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrCVUJjBO_02b_LySuchVNpEg1T9L77m-LGg&usqp=CAU", 
               "kidneybeans":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSb6FdYKjAcAuXi_MrswWx7RMMkNZDyHCc9FA&usqp=CAU", 
               "pigeonpeas":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdtLLbfIcLahMREYDQ2HQbt7zuW39XV2Yz-A&usqp=CAU", 
               "mothbeans":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJKtYf5xV5i3v04a2P89KThgo7-00sa3wGsA&usqp=CAU", 
               "mungbean":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRI74MWyeC1cRi1xhDsrvX5UwrEJAvk8xFsRg&usqp=CAU", 
               "blackgram":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTiV3ZTH2ZFkiIVqe39gj7LEj9TXjBOj5t5w&usqp=CAU", 
               "lentil":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0xIbantI8aAnWFPzIwIK8ByQ9TmLn2SxMsw&usqp=CAU", 
               "pomegranate":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHUwR3M_Te1-bwsdEL78N-YeYslMfSKHN4Hw&usqp=CAU", 
               "banana":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQctT7ej9S2SFwAGZm2StUHuxRp6AlCfTAs4w&usqp=CAU", 
               "mango":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT19Sdyr_QFejB7r0rV11s-yw99K78aoQWAug&usqp=CAU", 
               "grapes":"https://economictimes.indiatimes.com/thumb/msid-66115553,width-1200,height-900,resizemode-4,imgsize-961508/gettyimages-842928214.jpg?from=mdr", 
               "watermelon":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZ01wVt8E6i69R6V3IIuU0dACotBPgsP9uMQ&usqp=CAU", 
               "muskmelon":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwJJ8wlm7_F1AxE_858MGS3sPUKR6s7U2LUQ&usqp=CAU", 
               "apple":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQGzaJgtP-mhOw3l_pLv2_sOWlH1I04aoMX4Q&usqp=CAU", 
               "orange":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxxmS2hI2KjKaC18QvdSd3uOjdgrgGpU9E_g&usqp=CAU", 
               "papaya":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ52f04HS0tTWV_BOOXU2mU7ZAJdHUpwxcXfQ&usqp=CAU", 
               "coconut":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWWS_1K93iVkadnJfHkUq8LUSqCrLiFLkcFQ&usqp=CAU", 
               "cotton":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTXefpx4di21GIa5ot8RKDIJYfYaspy8AtnXw&usqp=CAU", 
               "jute":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRJcu8hWBFrXsMsCRrY7Y0XX9hJDeLFTDVWhQ&usqp=CAU", 
               "coffee":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOoW21Ps1t_AycwHtUWuh6YO-G9UHaa8EZnw&usqp=CAU"}
    return render_template('cropDetail.html', image_url1="{}".format(crop_url["rice"]), image1="{}".format("Rice"), image_url2="{}".format(crop_url["maize"]), image2="{}".format("maize"), image_url3="{}".format(crop_url["chickpea"]), image3="{}".format("chickpea"), image_url4="{}".format(crop_url["kidneybeans"]), image4="{}".format("kidneybeans"), image_url5="{}".format(crop_url["pigeonpeas"]), image5="{}".format("pigeonpeas"), image_url6="{}".format(crop_url["mothbeans"]), image6="{}".format("mothbeans"), image_url7="{}".format(crop_url["blackgram"]), image7="{}".format("blackgram"), image_url8="{}".format(crop_url["lentil"]), image8="{}".format("lentil"), image_url9="{}".format(crop_url["pomegranate"]), image9="{}".format("pomegranate"), image_url10="{}".format(crop_url["banana"]), image10="{}".format("banana"), image_url11="{}".format(crop_url["mango"]), image11="{}".format("mango"), image_url12="{}".format(crop_url["grapes"]), image12="{}".format("grapes"), image_url13="{}".format(crop_url["watermelon"]), image13="{}".format("watermelon"), image_url14="{}".format(crop_url["muskmelon"]), image14="{}".format("muskmelon"), image_url15="{}".format(crop_url["apple"]), image15="{}".format("apple"), image_url16="{}".format(crop_url["orange"]), image16="{}".format("orange"), image_url17="{}".format(crop_url["papaya"]), image17="{}".format("papaya"), image_url18="{}".format(crop_url["coconut"]), image18="{}".format("coconut"), image_url19="{}".format(crop_url["cotton"]), image19="{}".format("cotton"), image_url20="{}".format(crop_url["jute"]), image20="{}".format("jute"), image_url21="{}".format(crop_url["coffee"]), image21="{}".format("coffee"), image_url22="{}".format(crop_url["mungbean"]), image22="{}".format("mungbean"))

@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('crop_rec.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/myhome')
def myhome():
    return render_template('index.html')
@app.route('/rice')
def rice():
    return render_template('rice.html')
@app.route('/maize')
def maize():
    return render_template('maize.html')
@app.route('/coffee')
def coffee():
    return render_template('coffee.html')
@app.route('/orange')
def orange():
    return render_template('orange.html')
@app.route('/grapes')
def grapes():
    return render_template('grapes.html')
@app.route('/lentil')
def lentil():
    return render_template('lentil.html')
@app.route('/watermelon')
def watermelon():
    return render_template('watermelon.html')
@app.route('/muskmelon')
def muskmelon():
    return render_template('muskmelon.html')
@app.route('/kidneybeans')
def kidneybeans():
    return render_template('kidneybeans.html')
@app.route('/mothbeans')
def mothbeans():
    return render_template('mothbeans.html')
@app.route('/mungbeans')
def mungbeans():
    return render_template('mungbeans.html')
@app.route('/cotton')
def cotton():
    return render_template('cotton.html')
@app.route('/jute')
def jute():
    return render_template('jute.html')
@app.route('/coconut')
def coconut():
    return render_template('coconut.html')
@app.route('/pomogranate')
def pomogranate():
    return render_template('pomogranate.html')
@app.route('/chickpea')
def chickpea():
    return render_template('chickpea.html')
@app.route('/pegionpea')
def pegionpea():
    return render_template('pegionpea.html')
@app.route('/papaya')
def papaya():
    return render_template('papaya.html')
@app.route('/apple')
def apple():
    return render_template('apple.html')
@app.route('/mango')
def mango():
    return render_template('mango.html')
@app.route('/banana')
def banana():
    return render_template('banana.html')
@app.route('/blackgram')
def blackgram():
    return render_template('blackgram.html')




if __name__ == "__main__":
    app.run(debug=True)